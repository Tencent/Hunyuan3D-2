#include "rasterizer.h"
#include <fstream>

inline int pos2key(float* p, int resolution) {
    int x = (p[0] * 0.5 + 0.5) * resolution;
    int y = (p[1] * 0.5 + 0.5) * resolution;
    int z = (p[2] * 0.5 + 0.5) * resolution;
    return (x * resolution + y) * resolution + z;
}

inline void key2pos(int key, int resolution, float* p) {
    int x = key / resolution / resolution;
    int y = key / resolution % resolution;
    int z = key % resolution;
    p[0] = ((x + 0.5) / resolution - 0.5) * 2;
    p[1] = ((y + 0.5) / resolution - 0.5) * 2;
    p[2] = ((z + 0.5) / resolution - 0.5) * 2;
}

inline void key2cornerpos(int key, int resolution, float* p) {
    int x = key / resolution / resolution;
    int y = key / resolution % resolution;
    int z = key % resolution;
    p[0] = ((x + 0.75) / resolution - 0.5) * 2;
    p[1] = ((y + 0.25) / resolution - 0.5) * 2;
    p[2] = ((z + 0.75) / resolution - 0.5) * 2;
}

inline float* pos_ptr(int l, int i, int j, torch::Tensor t) {
    float* pdata = t.data_ptr<float>();
    int height = t.size(1);
    int width = t.size(2);
    return &pdata[((l * height + i) * width + j) * 4];
}

struct Grid
{
    std::vector<int> seq2oddcorner;
    std::vector<int> seq2evencorner;
    std::vector<int> seq2grid;
    std::vector<int> seq2normal;
    std::vector<int> seq2neighbor;
    std::unordered_map<int, int> grid2seq;
    std::vector<int> downsample_seq;
    int num_origin_seq;
    int resolution;
    int stride;
};

inline void pos_from_seq(Grid& grid, int seq, float* p) {
    auto k = grid.seq2grid[seq];
    key2pos(k, grid.resolution, p);
}

inline int fetch_seq(Grid& grid, int l, int i, int j, torch::Tensor pdata) {
    float* p = pos_ptr(l, i, j, pdata);
    if (p[3] == 0)
        return -1;
    auto key = pos2key(p, grid.resolution);
    int seq = grid.grid2seq[key];
    return seq;
}

inline int fetch_last_seq(Grid& grid, int i, int j, torch::Tensor pdata) {
    int num_layers = pdata.size(0);
    int l = 0;
    int idx = fetch_seq(grid, l, i, j, pdata);
    while (l < num_layers - 1) {
        l += 1;
        int new_idx = fetch_seq(grid, l, i, j, pdata);
        if (new_idx == -1)
            break;
        idx = new_idx;
    }
    return idx;
}

inline int fetch_nearest_seq(Grid& grid, int i, int j, int dim, float d, torch::Tensor pdata) {
    float p[3];
    float max_dist = 1e10;
    int best_idx = -1;
    int num_layers = pdata.size(0);
    for (int l = 0; l < num_layers; ++l) {
        int idx = fetch_seq(grid, l, i, j, pdata);
        if (idx == -1)
            break;
        pos_from_seq(grid, idx, p);
        float dist = std::abs(d - p[(dim + 2) % 3]);
        if (dist < max_dist) {
            max_dist = dist;
            best_idx = idx;
        }
    }
    return best_idx;
}

inline int fetch_nearest_seq_layer(Grid& grid, int i, int j, int dim, float d, torch::Tensor pdata) {
    float p[3];
    float max_dist = 1e10;
    int best_layer = -1;
    int num_layers = pdata.size(0);
    for (int l = 0; l < num_layers; ++l) {
        int idx = fetch_seq(grid, l, i, j, pdata);
        if (idx == -1)
            break;
        pos_from_seq(grid, idx, p);
        float dist = std::abs(d - p[(dim + 2) % 3]);
        if (dist < max_dist) {
            max_dist = dist;
            best_layer = l;
        }
    }
    return best_layer;
}

void FetchNeighbor(Grid& grid, int seq, float* pos, int dim, int boundary_info, std::vector<torch::Tensor>& view_layer_positions,
    int* output_indices)
{
    auto t = view_layer_positions[dim];
    int height = t.size(1);
    int width = t.size(2);
    int top = 0;
    int ci = 0;
    int cj = 0;
    if (dim == 0) {
        ci = (pos[1]/2+0.5)*height;
        cj = (pos[0]/2+0.5)*width;
    }
    else if (dim == 1) {
        ci = (pos[1]/2+0.5)*height;
        cj = (pos[2]/2+0.5)*width;
    }
    else {
        ci = (-pos[2]/2+0.5)*height;
        cj = (pos[0]/2+0.5)*width;
    }
    int stride = grid.stride;
    for (int ni = ci + stride; ni >= ci - stride; ni -= stride) {
        for (int nj = cj - stride; nj <= cj + stride; nj += stride) {
            int idx = -1;
            if (ni == ci && nj == cj)
                idx = seq;
            else if (!(ni < 0 || ni >= height || nj < 0 || nj >= width)) {
                if (boundary_info == -1)
                    idx = fetch_seq(grid, 0, ni, nj, t);
                else if (boundary_info == 1)
                    idx = fetch_last_seq(grid, ni, nj, t);
                else
                    idx = fetch_nearest_seq(grid, ni, nj, dim, pos[(dim + 2) % 3], t);
            }
            output_indices[top] = idx;
            top += 1;
        }
    }
}

void DownsampleGrid(Grid& src, Grid& tar)
{
    src.downsample_seq.resize(src.seq2grid.size(), -1);
    tar.resolution = src.resolution / 2;
    tar.stride = src.stride * 2;
    float pos[3];
    std::vector<int> seq2normal_count;
    for (int i = 0; i < src.seq2grid.size(); ++i) {
        key2pos(src.seq2grid[i], src.resolution, pos);
        int k = pos2key(pos, tar.resolution);
        int s = seq2normal_count.size();
        if (!tar.grid2seq.count(k)) {
            tar.grid2seq[k] = tar.seq2grid.size();
            tar.seq2grid.emplace_back(k);
            seq2normal_count.emplace_back(0);
            seq2normal_count.emplace_back(0);
            seq2normal_count.emplace_back(0);
            //tar.seq2normal.emplace_back(src.seq2normal[i]);
        } else {
            s = tar.grid2seq[k] * 3;
        }
        seq2normal_count[s + src.seq2normal[i]] += 1;
        src.downsample_seq[i] = tar.grid2seq[k];
    }
    tar.seq2normal.resize(seq2normal_count.size() / 3);
    for (int i = 0; i < seq2normal_count.size(); i += 3) {
        int t = 0;
        for (int j = 1; j < 3; ++j) {
            if (seq2normal_count[i + j] > seq2normal_count[i + t])
                t = j;
        }
        tar.seq2normal[i / 3] = t;
    }
}

void NeighborGrid(Grid& grid, std::vector<torch::Tensor> view_layer_positions, int v)
{
    grid.seq2evencorner.resize(grid.seq2grid.size(), 0);
    grid.seq2oddcorner.resize(grid.seq2grid.size(), 0);
    std::unordered_set<int> visited_seq;
    for (int vd = 0; vd < 3; ++vd) {
        auto t = view_layer_positions[vd];
        auto t0 = view_layer_positions[v];
        int height = t.size(1);
        int width = t.size(2);
        int num_layers = t.size(0);
        int num_view_layers = t0.size(0);
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                for (int l = 0; l < num_layers; ++l) {
                    int seq = fetch_seq(grid, l, i, j, t);
                    if (seq == -1)
                        break;
                    int dim = grid.seq2normal[seq];
                    if (dim != v)
                        continue;

                    float pos[3];
                    pos_from_seq(grid, seq, pos);

                    int ci = 0;
                    int cj = 0;
                    if (dim == 0) {
                        ci = (pos[1]/2+0.5)*height;
                        cj = (pos[0]/2+0.5)*width;
                    }
                    else if (dim == 1) {
                        ci = (pos[1]/2+0.5)*height;
                        cj = (pos[2]/2+0.5)*width;
                    }
                    else {
                        ci = (-pos[2]/2+0.5)*height;
                        cj = (pos[0]/2+0.5)*width;
                    }

                    if ((ci % (grid.stride * 2) < grid.stride) && (cj % (grid.stride * 2) >= grid.stride))
                        grid.seq2evencorner[seq] = 1;

                    if ((ci % (grid.stride * 2) >= grid.stride) && (cj % (grid.stride * 2) < grid.stride))
                        grid.seq2oddcorner[seq] = 1;

                    bool is_boundary = false;
                    if (vd == v) {
                        if (l == 0 || l == num_layers - 1)
                            is_boundary = true;
                        else {
                            int seq_new = fetch_seq(grid, l + 1, i, j, t);
                            if (seq_new == -1)
                                is_boundary = true;
                        }
                    }
                    int boundary_info = 0;
                    if (is_boundary && (l == 0))
                        boundary_info = -1;
                    else if (is_boundary)
                        boundary_info = 1;
                    if (visited_seq.count(seq))
                        continue;
                    visited_seq.insert(seq);

                    FetchNeighbor(grid, seq, pos, dim, boundary_info, view_layer_positions, &grid.seq2neighbor[seq * 9]);
                }
            }
        }
    }
}

void PadGrid(Grid& src, Grid& tar, std::vector<torch::Tensor>& view_layer_positions) {
    auto& downsample_seq = src.downsample_seq;
    auto& seq2evencorner = src.seq2evencorner;
    auto& seq2oddcorner = src.seq2oddcorner;
    int indices[9];
    std::vector<int> mapped_even_corners(tar.seq2grid.size(), 0);
    std::vector<int> mapped_odd_corners(tar.seq2grid.size(), 0);
    for (int i = 0; i < downsample_seq.size(); ++i) {
        if (seq2evencorner[i] > 0) {
            mapped_even_corners[downsample_seq[i]] = 1;
        }
        if (seq2oddcorner[i] > 0) {
            mapped_odd_corners[downsample_seq[i]] = 1;
        }
    }
    auto& tar_seq2normal = tar.seq2normal;
    auto& tar_seq2grid = tar.seq2grid;
    for (int i = 0; i < tar_seq2grid.size(); ++i) {
        if (mapped_even_corners[i] == 1 && mapped_odd_corners[i] == 1)
            continue;
        auto k = tar_seq2grid[i];
        float p[3];
        key2cornerpos(k, tar.resolution, p);

        int src_key = pos2key(p, src.resolution);
        if (!src.grid2seq.count(src_key)) {
            int seq = src.seq2grid.size();
            src.grid2seq[src_key] = seq;
            src.seq2evencorner.emplace_back((mapped_even_corners[i] == 0));
            src.seq2oddcorner.emplace_back((mapped_odd_corners[i] == 0));
            src.seq2grid.emplace_back(src_key);
            src.seq2normal.emplace_back(tar_seq2normal[i]);
            FetchNeighbor(src, seq, p, tar_seq2normal[i], 0, view_layer_positions, indices);
            for (int j = 0; j < 9; ++j) {
                src.seq2neighbor.emplace_back(indices[j]);
            }
            src.downsample_seq.emplace_back(i);
        } else {
            int seq = src.grid2seq[src_key];
            if (mapped_even_corners[i] == 0)
                src.seq2evencorner[seq] = 1;
            if (mapped_odd_corners[i] == 0)
                src.seq2oddcorner[seq] = 1;
        }
    }
}

std::vector<std::vector<torch::Tensor>> build_hierarchy(std::vector<torch::Tensor> view_layer_positions,
    std::vector<torch::Tensor> view_layer_normals, int num_level, int resolution)
{
    if (view_layer_positions.size() != 3 || num_level < 1) {
        printf("Alert! We require 3 layers and at least 1 level! (%d %d)\n", view_layer_positions.size(), num_level);
        return {{},{},{},{}};
    }

    std::vector<Grid> grids;
    grids.resize(num_level);

    std::vector<float> seq2pos;
    auto& seq2grid = grids[0].seq2grid;
    auto& seq2normal = grids[0].seq2normal;
    auto& grid2seq = grids[0].grid2seq;
    grids[0].resolution = resolution;
    grids[0].stride = 1;

    auto int64_options = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);

    for (int v = 0; v < 3; ++v) {
        int num_layers = view_layer_positions[v].size(0);
        int height = view_layer_positions[v].size(1);
        int width = view_layer_positions[v].size(2);
        float* data = view_layer_positions[v].data_ptr<float>();
        float* data_normal = view_layer_normals[v].data_ptr<float>();
        for (int l = 0; l < num_layers; ++l) {
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    float* p = &data[(i * width + j) * 4];
                    float* n = &data_normal[(i * width + j) * 3];
                    if (p[3] == 0)
                        continue;
                    auto k = pos2key(p, resolution);
                    if (!grid2seq.count(k)) {
                        int dim = 0;
                        for (int d = 0; d < 3; ++d) {
                            if (std::abs(n[d]) > std::abs(n[dim]))
                                dim = d;
                        }
                        dim = (dim + 1) % 3;
                        grid2seq[k] = seq2grid.size();
                        seq2grid.emplace_back(k);
                        seq2pos.push_back(p[0]);
                        seq2pos.push_back(p[1]);
                        seq2pos.push_back(p[2]);
                        seq2normal.emplace_back(dim);
                    }
                }
            }
            data += (height * width * 4);
            data_normal += (height * width * 3);
        }
    }

    for (int i = 0; i < num_level - 1; ++i) {
        DownsampleGrid(grids[i], grids[i + 1]);
    }

    for (int l = 0; l < num_level; ++l) {
        grids[l].seq2neighbor.resize(grids[l].seq2grid.size() * 9, -1);
        grids[l].num_origin_seq = grids[l].seq2grid.size();
        for (int d = 0; d < 3; ++d) {
            NeighborGrid(grids[l], view_layer_positions, d);
        }
    }

    for (int i = num_level - 2; i >= 0; --i) {
        PadGrid(grids[i], grids[i + 1], view_layer_positions);
    }
    for (int i = grids[0].num_origin_seq; i < grids[0].seq2grid.size(); ++i) {
        int k = grids[0].seq2grid[i];
        float p[3];
        key2pos(k, grids[0].resolution, p);
        seq2pos.push_back(p[0]);
        seq2pos.push_back(p[1]);
        seq2pos.push_back(p[2]);
    }

    std::vector<torch::Tensor> texture_positions(2);
    std::vector<torch::Tensor> grid_neighbors(grids.size());
    std::vector<torch::Tensor> grid_downsamples(grids.size() - 1);
    std::vector<torch::Tensor> grid_evencorners(grids.size());
    std::vector<torch::Tensor> grid_oddcorners(grids.size());

    texture_positions[0] = torch::zeros({seq2pos.size() / 3, 3}, float_options);
    texture_positions[1] = torch::zeros({seq2pos.size() / 3}, float_options);
    float* positions_out_ptr = texture_positions[0].data_ptr<float>();
    memcpy(positions_out_ptr, seq2pos.data(), sizeof(float) * seq2pos.size());
    positions_out_ptr = texture_positions[1].data_ptr<float>();
    for (int i = 0; i < grids[0].seq2grid.size(); ++i) {
        positions_out_ptr[i] = (i < grids[0].num_origin_seq);
    }

    for (int i = 0; i < grids.size(); ++i) {
        grid_neighbors[i] = torch::zeros({grids[i].seq2grid.size(), 9}, int64_options);
        long* nptr = grid_neighbors[i].data_ptr<long>();
        for (int j = 0; j < grids[i].seq2neighbor.size(); ++j) {
            nptr[j] = grids[i].seq2neighbor[j];
        }

        grid_evencorners[i] = torch::zeros({grids[i].seq2evencorner.size()}, int64_options);
        grid_oddcorners[i] = torch::zeros({grids[i].seq2oddcorner.size()}, int64_options);
        long* dptr = grid_evencorners[i].data_ptr<long>();
        for (int j = 0; j < grids[i].seq2evencorner.size(); ++j) {
            dptr[j] = grids[i].seq2evencorner[j];
        }
        dptr = grid_oddcorners[i].data_ptr<long>();
        for (int j = 0; j < grids[i].seq2oddcorner.size(); ++j) {
            dptr[j] = grids[i].seq2oddcorner[j];
        }            
        if (i + 1 < grids.size()) {
            grid_downsamples[i] = torch::zeros({grids[i].downsample_seq.size()}, int64_options);
            long* dptr = grid_downsamples[i].data_ptr<long>();
            for (int j = 0; j < grids[i].downsample_seq.size(); ++j) {
                dptr[j] = grids[i].downsample_seq[j];
            }
        }

    }
    return {texture_positions, grid_neighbors, grid_downsamples, grid_evencorners, grid_oddcorners};
}

std::vector<std::vector<torch::Tensor>> build_hierarchy_with_feat(
    std::vector<torch::Tensor> view_layer_positions,
    std::vector<torch::Tensor> view_layer_normals,
    std::vector<torch::Tensor> view_layer_feats,
    int num_level, int resolution)
{
    if (view_layer_positions.size() != 3 || num_level < 1) {
        printf("Alert! We require 3 layers and at least 1 level! (%d %d)\n", view_layer_positions.size(), num_level);
        return {{},{},{},{}};
    }

    std::vector<Grid> grids;
    grids.resize(num_level);

    std::vector<float> seq2pos;
    std::vector<float> seq2feat;
    auto& seq2grid = grids[0].seq2grid;
    auto& seq2normal = grids[0].seq2normal;
    auto& grid2seq = grids[0].grid2seq;
    grids[0].resolution = resolution;
    grids[0].stride = 1;

    auto int64_options = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);

    int feat_channel = 3;
    for (int v = 0; v < 3; ++v) {
        int num_layers = view_layer_positions[v].size(0);
        int height = view_layer_positions[v].size(1);
        int width = view_layer_positions[v].size(2);
        float* data = view_layer_positions[v].data_ptr<float>();
        float* data_normal = view_layer_normals[v].data_ptr<float>();
        float* data_feat = view_layer_feats[v].data_ptr<float>();
        feat_channel = view_layer_feats[v].size(3);
        for (int l = 0; l < num_layers; ++l) {
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    float* p = &data[(i * width + j) * 4];
                    float* n = &data_normal[(i * width + j) * 3];
                    float* f = &data_feat[(i * width + j) * feat_channel];
                    if (p[3] == 0)
                        continue;
                    auto k = pos2key(p, resolution);
                    if (!grid2seq.count(k)) {
                        int dim = 0;
                        for (int d = 0; d < 3; ++d) {
                            if (std::abs(n[d]) > std::abs(n[dim]))
                                dim = d;
                        }
                        dim = (dim + 1) % 3;
                        grid2seq[k] = seq2grid.size();
                        seq2grid.emplace_back(k);
                        seq2pos.push_back(p[0]);
                        seq2pos.push_back(p[1]);
                        seq2pos.push_back(p[2]);
                        for (int c = 0; c < feat_channel; ++c) {
                            seq2feat.emplace_back(f[c]);
                        }
                        seq2normal.emplace_back(dim);
                    }
                }
            }
            data += (height * width * 4);
            data_normal += (height * width * 3);
            data_feat += (height * width * feat_channel);
        }
    }

    for (int i = 0; i < num_level - 1; ++i) {
        DownsampleGrid(grids[i], grids[i + 1]);
    }

    for (int l = 0; l < num_level; ++l) {
        grids[l].seq2neighbor.resize(grids[l].seq2grid.size() * 9, -1);
        grids[l].num_origin_seq = grids[l].seq2grid.size();
        for (int d = 0; d < 3; ++d) {
            NeighborGrid(grids[l], view_layer_positions, d);
        }
    }

    for (int i = num_level - 2; i >= 0; --i) {
        PadGrid(grids[i], grids[i + 1], view_layer_positions);
    }
    for (int i = grids[0].num_origin_seq; i < grids[0].seq2grid.size(); ++i) {
        int k = grids[0].seq2grid[i];
        float p[3];
        key2pos(k, grids[0].resolution, p);
        seq2pos.push_back(p[0]);
        seq2pos.push_back(p[1]);
        seq2pos.push_back(p[2]);
        for (int c = 0; c < feat_channel; ++c) {
            seq2feat.emplace_back(0.5);
        }
    }

    std::vector<torch::Tensor> texture_positions(2);
    std::vector<torch::Tensor> texture_feats(1);
    std::vector<torch::Tensor> grid_neighbors(grids.size());
    std::vector<torch::Tensor> grid_downsamples(grids.size() - 1);
    std::vector<torch::Tensor> grid_evencorners(grids.size());
    std::vector<torch::Tensor> grid_oddcorners(grids.size());

    texture_positions[0] = torch::zeros({seq2pos.size() / 3, 3}, float_options);
    texture_positions[1] = torch::zeros({seq2pos.size() / 3}, float_options);
    texture_feats[0] = torch::zeros({seq2feat.size() / feat_channel, feat_channel}, float_options);
    float* positions_out_ptr = texture_positions[0].data_ptr<float>();
    memcpy(positions_out_ptr, seq2pos.data(), sizeof(float) * seq2pos.size());
    positions_out_ptr = texture_positions[1].data_ptr<float>();
    for (int i = 0; i < grids[0].seq2grid.size(); ++i) {
        positions_out_ptr[i] = (i < grids[0].num_origin_seq);
    }
    float* feats_out_ptr = texture_feats[0].data_ptr<float>();
    memcpy(feats_out_ptr, seq2feat.data(), sizeof(float) * seq2feat.size());

    for (int i = 0; i < grids.size(); ++i) {
        grid_neighbors[i] = torch::zeros({grids[i].seq2grid.size(), 9}, int64_options);
        long* nptr = grid_neighbors[i].data_ptr<long>();
        for (int j = 0; j < grids[i].seq2neighbor.size(); ++j) {
            nptr[j] = grids[i].seq2neighbor[j];
        }
        grid_evencorners[i] = torch::zeros({grids[i].seq2evencorner.size()}, int64_options);
        grid_oddcorners[i] = torch::zeros({grids[i].seq2oddcorner.size()}, int64_options);
        long* dptr = grid_evencorners[i].data_ptr<long>();
        for (int j = 0; j < grids[i].seq2evencorner.size(); ++j) {
            dptr[j] = grids[i].seq2evencorner[j];
        }
        dptr = grid_oddcorners[i].data_ptr<long>();
        for (int j = 0; j < grids[i].seq2oddcorner.size(); ++j) {
            dptr[j] = grids[i].seq2oddcorner[j];
        }
        if (i + 1 < grids.size()) {
            grid_downsamples[i] = torch::zeros({grids[i].downsample_seq.size()}, int64_options);
            long* dptr = grid_downsamples[i].data_ptr<long>();
            for (int j = 0; j < grids[i].downsample_seq.size(); ++j) {
                dptr[j] = grids[i].downsample_seq[j];
            }
        }
    }
    return {texture_positions, texture_feats, grid_neighbors, grid_downsamples, grid_evencorners, grid_oddcorners};
}
