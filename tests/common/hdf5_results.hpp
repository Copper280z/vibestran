#pragma once
// Independent readers for MAGSTAT result files, written against the HDF5 C
// API rather than the solver's writer, for tests only.
#include <cmath>
#include <filesystem>
#include <hdf5.h>
#include <string>
#include <vector>

namespace vibestran::test {
struct Table {
  std::vector<double> values; // row-major; HDF5 converts integer storage
  std::vector<hsize_t> dims;
  [[nodiscard]] double at(size_t row, size_t column = 0) const {
    return values[row * (dims.size() > 1 ? dims[1] : 1) + column];
  }
  [[nodiscard]] size_t rows() const { return dims.empty() ? 0 : dims[0]; }
};

// True when every component of the path exists.
inline bool h5_exists(const std::filesystem::path &file,
                      const std::string &path) {
  const hid_t f = H5Fopen(file.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (f < 0)
    return false;
  bool found = true;
  for (size_t at = path.find('/', 1); found; at = path.find('/', at + 1)) {
    found = H5Lexists(f, path.substr(0, at).c_str(), H5P_DEFAULT) > 0;
    if (at == std::string::npos)
      break;
  }
  H5Fclose(f);
  return found;
}

// A whole dataset as doubles; empty when it does not exist.
inline Table h5_table(const std::filesystem::path &file,
                      const std::string &path) {
  Table t;
  const hid_t f = H5Fopen(file.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  const hid_t d = H5Dopen2(f, path.c_str(), H5P_DEFAULT);
  if (d >= 0) {
    const hid_t space = H5Dget_space(d);
    t.dims.resize(static_cast<size_t>(H5Sget_simple_extent_ndims(space)));
    H5Sget_simple_extent_dims(space, t.dims.data(), nullptr);
    t.values.resize(static_cast<size_t>(H5Sget_simple_extent_npoints(space)));
    if (!t.values.empty())
      H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              t.values.data());
    H5Sclose(space);
    H5Dclose(d);
  }
  H5Fclose(f);
  return t;
}

// A numeric scalar attribute as a double; NaN when it does not exist.
inline double h5_attribute(const std::filesystem::path &file,
                           const std::string &path, const std::string &name) {
  const hid_t f = H5Fopen(file.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  double value = std::nan("");
  const hid_t a = H5Aopen_by_name(f, path.c_str(), name.c_str(), H5P_DEFAULT,
                                  H5P_DEFAULT);
  if (a >= 0) {
    H5Aread(a, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(a);
  }
  H5Fclose(f);
  return value;
}

// Names of a group's links, in creation order.
inline std::vector<std::string> h5_children(const std::filesystem::path &file,
                                            const std::string &path) {
  std::vector<std::string> names;
  const hid_t f = H5Fopen(file.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  const hid_t g = H5Gopen2(f, path.c_str(), H5P_DEFAULT);
  H5G_info_t info{};
  if (g >= 0 && H5Gget_info(g, &info) >= 0)
    for (hsize_t i = 0; i < info.nlinks; ++i) {
      const auto length = H5Lget_name_by_idx(g, ".", H5_INDEX_CRT_ORDER,
                                             H5_ITER_INC, i, nullptr, 0,
                                             H5P_DEFAULT);
      std::string name(length > 0 ? static_cast<size_t>(length) : 0, '\0');
      if (length > 0)
        H5Lget_name_by_idx(g, ".", H5_INDEX_CRT_ORDER, H5_ITER_INC, i,
                           name.data(), name.size() + 1, H5P_DEFAULT);
      names.push_back(name);
    }
  if (g >= 0)
    H5Gclose(g);
  H5Fclose(f);
  return names;
}
} // namespace vibestran::test
