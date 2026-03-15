# meson.build — NastranSolver build system
# Requires: Meson >= 1.0, C++20 compiler, Eigen3
# Usage:
#   meson setup build --buildtype=release
#   cd build && meson compile
#   meson test

project('nastran_solver', 'cpp',
  version : '0.1.0',
  default_options : [
    'cpp_std=c++20',
    'warning_level=3',
    'b_ndebug=if-release',
    'default_library=static',
  ]
)

# ── Compiler setup ────────────────────────────────────────────────────────────
cxx = meson.get_compiler('cpp')

# Compiler-specific flags
extra_args = []
if cxx.get_id() == 'gcc' or cxx.get_id() == 'clang'
  extra_args += ['-Wno-unused-parameter', '-Wfatal-errors']
  # Enable AVX2 for release builds (benefits Eigen)
  if get_option('buildtype') == 'release'
    extra_args += ['-march=native', '-ffast-math']
  endif
elif cxx.get_id() == 'msvc'
  extra_args += ['/wd4100', '/EHsc']
  if get_option('buildtype') == 'release'
    extra_args += ['/arch:AVX2', '/fp:fast']
  endif
endif

add_project_arguments(extra_args, language : 'cpp')

# ── Eigen dependency ──────────────────────────────────────────────────────────
# First try pkg-config, then fall back to third_party/ bundled headers
eigen_dep = dependency('eigen3',
  required : false,
  fallback : ['eigen3', 'eigen_dep'],
)

if not eigen_dep.found()
  # NOT CURRENTLY IMPLEMENTED
  # Use bundled Eigen headers in third_party/eigen
  # eigen_inc = include_directories('third_party/eigen')
  # eigen_dep = declare_dependency(include_directories : eigen_inc)
  message('Using bundled Eigen from third_party/')
  error('Bundled Eigen not currently implemented')
else
  message('Using system Eigen3')
endif

# ── Thread support (for future parallel assembly) ────────────────────────────
thread_dep = dependency('threads')

# ── cppcheck ─────────────────────────────────────────────────────────────────
cppcheck = find_program('cppcheck', required: false)
if cppcheck.found()
  run_target('cppcheck',
    command: [
      cppcheck,
      '--project=' + join_paths(meson.project_build_root(), 'compile_commands.json'),
      '--enable=all',
      '--inline-suppr',
      '--quiet',
      '--suppress=missingIncludeSystem',
      '--suppress=checkersReport',
      '--quiet',
      # Optional: Add extra arguments
      # '--force', 
      # '-j', '4',
    ]
  )
endif

# ── Source files ─────────────────────────────────────────────────────────────
solver_sources = files(
  # Core
  'src/core/model.cpp',
  'src/core/dof_map.cpp',
  'src/core/sparse_matrix.cpp',
  # Elements
  'src/elements/cquad4.cpp',
  'src/elements/ctria3.cpp',
  'src/elements/solid_elements.cpp',
  'src/elements/element_factory.cpp',
  # I/O
  'src/io/bdf_parser.cpp',
  'src/io/results.cpp',
  # Solver
  'src/solver/solver_backend.cpp',
  'src/solver/linear_static.cpp',
)

# Include directories
inc = include_directories('include')

# ── Library (solver logic, no main) ──────────────────────────────────────────
solver_lib = static_library(
  'nastransolver',
  solver_sources,
  include_directories : inc,
  dependencies        : [eigen_dep, thread_dep],
  install             : false,
)

solver_lib_dep = declare_dependency(
  link_with           : solver_lib,
  include_directories : inc,
  dependencies        : [eigen_dep, thread_dep],
)

# ── Executable ───────────────────────────────────────────────────────────────
nastran_exe = executable(
  'nastran_solver',
  'src/main.cpp',
  dependencies : solver_lib_dep,
  install      : true,
)

# ── Tests ────────────────────────────────────────────────────────────────────
# Google Test
gtest_dep = dependency('gtest', main : true, fallback : ['gtest', 'gtest_dep'])

# Unit tests
unit_test_sources = files(
  'tests/unit/test_dof_map.cpp',
  'tests/unit/test_bdf_parser.cpp',
  'tests/unit/test_elements.cpp',
  'tests/unit/test_types.cpp',
  'tests/unit/test_sparse_matrix.cpp',
)

unit_tests = executable(
  'unit_tests',
  unit_test_sources,
  dependencies : [solver_lib_dep, gtest_dep],
  include_directories : inc,
)

test('unit_tests', unit_tests,
  timeout : 60,
  suite   : 'unit',
)

# Integration tests
integration_test_sources = files(
  'tests/integration/test_integration.cpp',
)

integration_tests = executable(
  'integration_tests',
  integration_test_sources,
  dependencies : [solver_lib_dep, gtest_dep],
  include_directories : inc,
)

test('integration_tests', integration_tests,
  timeout : 120,
  suite   : 'integration',
  # Integration tests are more numerically sensitive; run with verbose output
  args    : ['--gtest_output=xml:' + meson.project_build_root() + '/integration_test_results.xml'],
)

# ── Summary ───────────────────────────────────────────────────────────────────
summary({
  'Build type'  : get_option('buildtype'),
  'C++ std'     : 'C++20',
  'Eigen'       : eigen_dep.found(),
  'Solver lib'  : solver_lib.name(),
  'Executable'  : nastran_exe.name(),
}, section : 'NastranSolver')
