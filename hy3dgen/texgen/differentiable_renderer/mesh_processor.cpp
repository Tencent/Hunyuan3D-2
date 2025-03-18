#include <vector>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <chrono>
#include <iostream>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

// Define version string
// 1.0 = smoothing only, 1.1 = smoothing + inpainting
const string MODULE_VERSION = "1.1";

std::pair<py::array_t<float>, py::array_t<uint8_t>> meshVerticeInpaint_smooth(
    py::array_t<float> texture, py::array_t<uint8_t> mask,
    py::array_t<float> vtx_pos, py::array_t<float> vtx_uv,
    py::array_t<int> pos_idx, py::array_t<int> uv_idx) {

    auto start_total = chrono::high_resolution_clock::now();

    // Buffer setup
    auto texture_buf = texture.request();
    auto mask_buf = mask.request();
    auto vtx_pos_buf = vtx_pos.request();
    auto vtx_uv_buf = vtx_uv.request();
    auto pos_idx_buf = pos_idx.request();
    auto uv_idx_buf = uv_idx.request();

    int texture_height = texture_buf.shape[0];
    int texture_width = texture_buf.shape[1];
    int texture_channel = texture_buf.shape[2];
    float* texture_ptr = static_cast<float*>(texture_buf.ptr);
    uint8_t* mask_ptr = static_cast<uint8_t*>(mask_buf.ptr);

    int vtx_num = vtx_pos_buf.shape[0];
    float* vtx_pos_ptr = static_cast<float*>(vtx_pos_buf.ptr);
    float* vtx_uv_ptr = static_cast<float*>(vtx_uv_buf.ptr);
    int* pos_idx_ptr = static_cast<int*>(pos_idx_buf.ptr);
    int* uv_idx_ptr = static_cast<int*>(uv_idx_buf.ptr);

    vector<uint8_t> vtx_mask(vtx_num, 0);
    vector<vector<float>> vtx_color(vtx_num, vector<float>(texture_channel));
    unordered_set<int> uncolored_vtxs;
    vector<vector<int>> G(vtx_num);

    // Initial setup
    auto start_setup = chrono::high_resolution_clock::now();
    vector<pair<int, int>> uv_coords(vtx_num);
    int colored_pixels = 0;
    float sample_color[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < uv_idx_buf.shape[0]; ++i) {
        for (int k = 0; k < 3; ++k) {
            int vtx_uv_idx = uv_idx_ptr[i * 3 + k];
            int vtx_idx = pos_idx_ptr[i * 3 + k];
            uv_coords[vtx_idx] = {
                static_cast<int>(round(vtx_uv_ptr[vtx_uv_idx * 2] * (texture_width - 1))),
                static_cast<int>(round((1.0 - vtx_uv_ptr[vtx_uv_idx * 2 + 1]) * (texture_height - 1)))
            };
            int uv_v = uv_coords[vtx_idx].first;
            int uv_u = uv_coords[vtx_idx].second;
            int pixel_idx = uv_u * texture_width + uv_v;

            if (uv_u >= 0 && uv_u < texture_height && uv_v >= 0 && uv_v < texture_width && mask_ptr[pixel_idx] > 0) {
                vtx_mask[vtx_idx] = 1;
                for (int c = 0; c < texture_channel; ++c) {
                    vtx_color[vtx_idx][c] = texture_ptr[pixel_idx * texture_channel + c];
                    if (colored_pixels == 0) sample_color[c] = vtx_color[vtx_idx][c];
                }
                colored_pixels++;
            } else {
                uncolored_vtxs.insert(vtx_idx);
            }
            G[vtx_idx].push_back(pos_idx_ptr[i * 3 + (k + 1) % 3]);
        }
    }
    auto end_setup = chrono::high_resolution_clock::now();

    // Smoothing
    auto start_smoothing = chrono::high_resolution_clock::now();
    const int thread_count = thread::hardware_concurrency();
    vector<thread> threads;
    mutex mtx;
    int max_iterations = 5;

    auto smooth_vertices = [&](int start_idx, int end_idx) {
        vector<float> sum_color(texture_channel);
        for (int iter = 0; iter < max_iterations && !uncolored_vtxs.empty(); ++iter) {
            vector<int> to_color;
            {
                lock_guard<mutex> lock(mtx);
                to_color.assign(uncolored_vtxs.begin(), uncolored_vtxs.end());
            }
            for (int vtx_idx : to_color) {
                fill(sum_color.begin(), sum_color.end(), 0.0f);
                float total_weight = 0.0f;
                float vtx_0[3] = {vtx_pos_ptr[vtx_idx * 3],
                                vtx_pos_ptr[vtx_idx * 3 + 1],
                                vtx_pos_ptr[vtx_idx * 3 + 2]};
                for (int connected_idx : G[vtx_idx]) {
                    if (vtx_mask[connected_idx]) {
                        float vtx1[3] = {vtx_pos_ptr[connected_idx * 3],
                                       vtx_pos_ptr[connected_idx * 3 + 1],
                                       vtx_pos_ptr[connected_idx * 3 + 2]};
                        float dx = vtx_0[0] - vtx1[0];
                        float dy = vtx_0[1] - vtx1[1];
                        float dz = vtx_0[2] - vtx1[2];
                        float dist_weight = 1.0f / max(sqrt(dx*dx + dy*dy + dz*dz), 1E-4f);
                        dist_weight *= dist_weight;
                        for (int c = 0; c < texture_channel; ++c) {
                            sum_color[c] += vtx_color[connected_idx][c] * dist_weight;
                        }
                        total_weight += dist_weight;
                    }
                }
                if (total_weight > 0.0f) {
                    lock_guard<mutex> lock(mtx);
                    for (int c = 0; c < texture_channel; ++c) {
                        vtx_color[vtx_idx][c] = sum_color[c] / total_weight;
                    }
                    vtx_mask[vtx_idx] = 1;
                    uncolored_vtxs.erase(vtx_idx);
                }
            }
        }
    };

    int chunk_size = vtx_num / thread_count;
    for (int t = 0; t < thread_count; ++t) {
        int start = t * chunk_size;
        int end = (t == thread_count - 1) ? vtx_num : start + chunk_size;
        threads.emplace_back(smooth_vertices, start, end);
    }
    for (auto& t : threads) t.join();
    auto end_smoothing = chrono::high_resolution_clock::now();

    // Output preparation
    auto start_output = chrono::high_resolution_clock::now();
    py::array_t<float> new_texture({texture_height, texture_width, texture_channel});
    py::array_t<uint8_t> new_mask({texture_height, texture_width});
    auto new_texture_buf = new_texture.request();
    auto new_mask_buf = new_mask.request();
    float* new_texture_ptr = static_cast<float*>(new_texture_buf.ptr);
    uint8_t* new_mask_ptr = static_cast<uint8_t*>(new_mask_buf.ptr);

    memcpy(new_texture_ptr, texture_ptr, texture_buf.size * sizeof(float));
    memcpy(new_mask_ptr, mask_ptr, mask_buf.size * sizeof(uint8_t));

    int updated_pixels = 0;
    float sample_updated_color[3] = {0.0f, 0.0f, 0.0f};
    for (int face_idx = 0; face_idx < uv_idx_buf.shape[0]; ++face_idx) {
        for (int k = 0; k < 3; ++k) {
            int vtx_idx = pos_idx_ptr[face_idx * 3 + k];
            if (vtx_mask[vtx_idx]) {
                int uv_v = uv_coords[vtx_idx].first;
                int uv_u = uv_coords[vtx_idx].second;
                if (uv_u >= 0 && uv_u < texture_height && uv_v >= 0 && uv_v < texture_width) {
                    int pixel_idx = uv_u * texture_width + uv_v;
                    for (int c = 0; c < texture_channel; ++c) {
                        new_texture_ptr[pixel_idx * texture_channel + c] = vtx_color[vtx_idx][c];
                        if (updated_pixels == 0) sample_updated_color[c] = vtx_color[vtx_idx][c];
                    }
                    new_mask_ptr[pixel_idx] = 255;
                    updated_pixels++;
                }
            }
        }
    }
    auto end_output = chrono::high_resolution_clock::now();

    // Custom inpainting
    auto start_inpaint = chrono::high_resolution_clock::now();
    vector<thread> inpaint_threads;
    int inpainted_pixels = 0;
    float sample_inpaint_color[3] = {0.0f, 0.0f, 0.0f};
    auto inpaint_region = [&](int start_y, int end_y) {
        int local_inpainted = 0;
        for (int y = start_y; y < end_y; ++y) {
            for (int x = 0; x < texture_width; ++x) {
                int idx = y * texture_width + x;
                if (new_mask_ptr[idx] != 255) {  // Uncolored pixel
                    float sum_color[3] = {0.0f, 0.0f, 0.0f};
                    float total_weight = 0.0f;
                    int kernel_size = 3;  // 7x7 neighborhood
                    for (int dy = -kernel_size; dy <= kernel_size; ++dy) {
                        for (int dx = -kernel_size; dx <= kernel_size; ++dx) {
                            int ny = y + dy;
                            int nx = x + dx;
                            if (ny >= 0 && ny < texture_height && nx >= 0 && nx < texture_width) {
                                int n_idx = ny * texture_width + nx;
                                if (new_mask_ptr[n_idx] == 255) {
                                    float dist = sqrt(static_cast<float>(dx*dx + dy*dy));
                                    float weight = 1.0f / max(dist, 1E-4f);
                                    for (int c = 0; c < texture_channel; ++c) {
                                        sum_color[c] += new_texture_ptr[n_idx * texture_channel + c] * weight;
                                    }
                                    total_weight += weight;
                                }
                            }
                        }
                    }
                    if (total_weight > 0.0f) {
                        for (int c = 0; c < texture_channel; ++c) {
                            new_texture_ptr[idx * texture_channel + c] = sum_color[c] / total_weight;
                            if (local_inpainted == 0) sample_inpaint_color[c] = sum_color[c] / total_weight;
                        }
                        new_mask_ptr[idx] = 255;
                        local_inpainted++;
                    }
                }
            }
        }
        lock_guard<mutex> lock(mtx);
        inpainted_pixels += local_inpainted;
    };

    int inpaint_chunk_size = texture_height / thread_count;
    for (int t = 0; t < thread_count; ++t) {
        int start = t * inpaint_chunk_size;
        int end = (t == thread_count - 1) ? texture_height : start + inpaint_chunk_size;
        inpaint_threads.emplace_back(inpaint_region, start, end);
    }
    for (auto& t : inpaint_threads) t.join();
    auto end_inpaint = chrono::high_resolution_clock::now();

    return {new_texture, new_mask};
}

std::pair<py::array_t<float>, py::array_t<uint8_t>> meshVerticeInpaint(
    py::array_t<float> texture, py::array_t<uint8_t> mask,
    py::array_t<float> vtx_pos, py::array_t<float> vtx_uv,
    py::array_t<int> pos_idx, py::array_t<int> uv_idx,
    const std::string& method = "smooth") {
    if (method == "smooth") {
        return meshVerticeInpaint_smooth(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx);
    } else {
        throw std::invalid_argument("Invalid method. Use 'smooth' or 'forward'.");
    }
}

// Function to return the module version
string get_module_version() {
    return MODULE_VERSION;
}

PYBIND11_MODULE(mesh_processor, m) {
    m.def("meshVerticeInpaint", &meshVerticeInpaint, "A function to process mesh",
          py::arg("texture"), py::arg("mask"),
          py::arg("vtx_pos"), py::arg("vtx_uv"),
          py::arg("pos_idx"), py::arg("uv_idx"),
          py::arg("method") = "smooth");
    m.def("get_module_version", &get_module_version, "Get the version of the mesh_processor module");
}