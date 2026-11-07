# The vendored supernodal LDL^T library; see README.md. Its orderings are required: without
# METIS a solid factors 2x to 3x slower, and the magnetics need AMD (FactorOptions).
ldlt_cc = meson.get_compiler('c')
ldlt_deps = []
ldlt_defs = []
ldlt_sources = files(
  'src/util.c',
  'src/perm.c',
  'src/etree.c',
  'src/symbolic.c',
  'src/ordering.c',
  'src/factor.c',
  'src/factor_kernels.c',
  'src/factor_pivot.c',
  'src/solve.c',
  'src/io_mm.c',
)

# ── BLAS ──
if have_accelerate
  ldlt_deps += accelerate_dep
  ldlt_defs += ['-DLDLT_USE_SYSTEM_BLAS=1', '-DLDLT_BLAS_ACCELERATE=1', '-DLDLT_HAVE_LAPACK=1']
else
  _ldlt_blas = dependency('', required : false)
  # OpenBLAS first: it is the one whose threads the library can manage.
  foreach _name : ['openblas', 'cblas', 'blas']
    if not _ldlt_blas.found()
      _candidate = dependency(_name, required : false)
      if _candidate.found() and ldlt_cc.has_header('cblas.h', dependencies : _candidate) and ldlt_cc.has_function('cblas_dgemm', dependencies : _candidate)
        _ldlt_blas = _candidate
      endif
    endif
  endforeach
  if _ldlt_blas.found()
    ldlt_deps += _ldlt_blas
    _lapack = dependency('lapack', required : false)
    if _lapack.found()
      ldlt_deps += _lapack
    endif
    ldlt_defs += '-DLDLT_USE_SYSTEM_BLAS=1'
    if ldlt_cc.has_function('dpotrf_', dependencies : ldlt_deps)
      ldlt_defs += '-DLDLT_HAVE_LAPACK=1'
    endif
    if ldlt_cc.has_function('openblas_set_num_threads', dependencies : ldlt_deps)
      ldlt_defs += '-DLDLT_HAVE_OPENBLAS=1'
    endif
  else
    warning('No system BLAS: the LDL^T backend uses its reference BLAS, which is far slower. ' +
            'Install libopenblas-dev.')
    ldlt_defs += '-DLDLT_USE_SYSTEM_BLAS=0'
    ldlt_sources += files('src/blas_ref.c')
  endif
endif

# ── Orderings ──
_ldlt_header_dirs = []
foreach _d : ['/opt/homebrew/include', '/opt/homebrew/include/suitesparse', '/usr/local/include',
              '/usr/local/include/suitesparse', '/usr/include/suitesparse']
  if fs.is_dir(_d)
    _ldlt_header_dirs += _d
  endif
endforeach
_ldlt_lib_dirs = []
foreach _d : ['/opt/homebrew/lib', '/usr/local/lib']
  if fs.is_dir(_d)
    _ldlt_lib_dirs += _d
  endif
endforeach
foreach _name : ['amd', 'metis']
  _dep = dependency(_name, required : false)
  if not _dep.found()
    _lib = ldlt_cc.find_library(_name, dirs : _ldlt_lib_dirs, required : false)
    if _lib.found() and ldlt_cc.has_header(_name + '.h')
      _dep = _lib
    elif _lib.found()
      foreach _d : _ldlt_header_dirs
        if not _dep.found() and ldlt_cc.has_header(_name + '.h', args : ['-I' + _d])
          _dep = declare_dependency(dependencies : _lib,
                                    include_directories : include_directories(_d, is_system : true))
        endif
      endforeach
    endif
  endif
  if not _dep.found()
    error('The LDL^T backend needs ' + _name + '. Install libmetis-dev and libsuitesparse-dev ' +
          '(Linux) or metis and suite-sparse (Homebrew).')
  endif
  ldlt_deps += _dep
  ldlt_defs += '-DLDLT_HAVE_' + _name.to_upper() + '=1'
endforeach

# ── Threads ──
_ldlt_omp = dependency('openmp', language : 'c', required : false)
if _ldlt_omp.found()
  ldlt_deps += _ldlt_omp
  ldlt_defs += '-DLDLT_HAVE_OPENMP=1'
endif

if host_machine.system() != 'windows'
  ldlt_defs += '-DLDLT_USE_POSIX_SIGNALS=1'
  if host_machine.system() != 'darwin'
    ldlt_defs += '-D_POSIX_C_SOURCE=200809L'
  endif
endif

_ldlt_tuning = []
if get_option('buildtype') == 'release'
  foreach _a : ['-march=native', '-funroll-loops']
    if ldlt_cc.has_argument(_a)
      _ldlt_tuning += _a
    endif
  endforeach
endif
if have_accelerate
  _ldlt_tuning += '-Wno-deprecated-declarations'
endif

ldlt_lib = static_library(
  'ldlt',
  ldlt_sources,
  include_directories : include_directories('include', 'src'),
  c_args : ldlt_defs + _ldlt_tuning,
  dependencies : ldlt_deps,
  override_options : ['c_std=c11', 'warning_level=2'],
)

ldlt_dep = declare_dependency(
  link_with : ldlt_lib,
  include_directories : include_directories('include'),
  dependencies : ldlt_deps,
)
