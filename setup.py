
#!/usr/bin/env python

################################################################################
# All the control parameters should go here

source_directory_list = ['pycpx']
compiler_args = []
link_args = []
version = "0.03"
description="A fast and flexible numpy-based wrapper for CPLex's Optimization Suite."
author = "Hoyt Koepke"
author_email="hoytak@gmail.com"
name = 'pycpx'
scripts = []
url = "http://www.stat.washington.edu/~hoytak/code/pycpx/"
download_url = "http://pypi.python.org/packages/source/p/pycpx/pycpx-0.02.tar.gz"

long_description = \
"""
PyCPX is a python wrapper for the CPlex Optimization Suite that
focuses on speed, ease of use, and seamless integration with numpy.
CPlex is a powerful solver for linear and quadratic programs over
real, linear, and boolean variables.  PyCPX allows one to naturally
express such programs using numpy and natural python constructs.

PyCPX requires IBM's ILog Concert Technology Suite, which is available
for free under IBM's Academic Initiative program or as part of the
CPlex Optimization Suite.

Version 0.03 fixes an error in array slicing.

Version 0.02 fixes several small bugs and gives vast speed
improvements for model creation in many models.
"""

classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Cython',
    'Programming Language :: C++',
    'Topic :: Scientific/Engineering',
    ]

# Stuff for extension module stuff
extra_library_dirs = []
extra_include_dirs = []

# Need to find cplex together 

library_includes = ['concert', 'ilocplex', 'cplex', 'm']
specific_libraries = {}

compiler_args += ['-DIL_STD', '-pthread']

################################################################################
# Shouldn't have to adjust anything below this line...

from glob import glob
import os
from os.path import split, join
from itertools import chain, product
import sys

import numpy
extra_include_dirs += [numpy.get_include()]


from distutils.core import setup
from distutils.extension import Extension

################################################################################
# Find cplex!

search_required_include_files = [
    "ilconcert/iloexpression.h",
    "ilconcert/iloalg.h",
    "ilconcert/iloenv.h",
    "ilconcert/ilosolution.h"]

search_required_concert_lib_files = ["libconcert.a", "libconcert.so", "libconcert.dll"]
search_required_cplex_lib_files = ["libcplex.a", "libcplex.so", "libcplex.dll"]

def process_path(s):
    return [p for p in s.split(';') if p]

################################################################################
# Figure out whether we're 64bit or 32bit

import platform
bits = platform.architecture()[0]

print "\n##################################################"
print "Figuring out CPlex setup.\n"
print "Targeting %s platform." % bits

is64bit = "64" in bits

# Get environment path definitions
search_paths = []

if "CPLEX_PATH" in os.environ:
    search_paths += process_path(os.environ["CPLEX_PATH"])

search_paths += ['/usr/ilog/']

if "INCLUDE_PATH" in os.environ:
    search_paths += process_path(os.environ["PATH"])
    
if "PATH" in os.environ:
    search_paths += process_path(os.environ["PATH"])

# Now find CPlex
def attempt_path_find(p, mid_level, goal_list, mode, path_filter, test_64bit):

    ret_set = set()

    matching_files = set(chain(*[glob(join(p, mid_level, l)) for l in goal_list]))
    if ((mode == 'and' and len(matching_files) >= len(goal_list))
        or mode == 'or' and len(matching_files) >= 1):
        
        for fp, gf in product(matching_files, goal_list):
            if gf in fp:
                ret_set.add(fp[:-len(gf)])

    return [r for r in ret_set if
            ((not test_64bit or not is64bit or "64" in r) and path_filter in r)]

# Find the include path
def find_path(pl, goal_list, name, mode, path_filter, test_64bit):
    for p in pl:
        for mid_level in ['*', '*/*', '*/*/*', '*/*/*/*', '*/*/*/*/*', '../*', '../*/*']:
            
            candidates = attempt_path_find(p, mid_level, goal_list, mode, path_filter, test_64bit)

            if candidates:
                return list(candidates)
        
    raise Exception(("CPLEX %s not found: please set environment variable CPLEX_PATH "
                     "to point to the base of the CPlex/Concert installation. "
                     "Attempting to find files: %s." )
                    % (name, (', '.join(goal_list))))

include_path = find_path(search_paths, search_required_include_files,
                         "concert include directory", 'and', '', False)
print "Using CPLEX concert include directory(s): %s" % (",".join(include_path))
extra_include_dirs += include_path

cplex_lib_path = find_path(search_paths, search_required_cplex_lib_files,
                           "cplex library", 'or', '', True)
print "Using CPLEX library directory(s): %s" % (",".join(cplex_lib_path))
extra_library_dirs += cplex_lib_path

concert_lib_path = find_path(search_paths, search_required_concert_lib_files,
                             "concert library", 'or', 'concert', True)
print "Using CPLEX concert library directory(s): %s" % (",".join(concert_lib_path))
extra_library_dirs += concert_lib_path

######################################################
# First have to see if we're authorized to use cython files, or if we
# should instead compile the included files

if "--cython" in sys.argv:
    cython_mode = True
    del sys.argv[sys.argv.index("--cython")]
else:
    cython_mode = False

if "--debug" in sys.argv:
    debug_mode_c_code = True
    del sys.argv[sys.argv.index("--debug")]
else:
    debug_mode_c_code = False

# Get all the cython files in the sub directories and in this directory
if cython_mode:
    cython_files = dict( (d, glob(join(d, "*.pyx"))) for d in source_directory_list + ['.'])
else:
    cython_files = {}

all_cython_files = set(chain(*cython_files.values()))

print "+++++++++++++++++++"

if cython_mode:
    print "Cython Files Found: \n%s\n+++++++++++++++++++++" % ", ".join(sorted(all_cython_files))
else:
    print "Cython support disabled; compiling extensions from pregenerated C sources."
    print "To enable cython, run setup.py with the option --cython."
    print "+++++++++++++++++++"

# Set the compiler arguments -- Add in the environment path stuff
ld_library_path = os.getenv("LD_LIBRARY_PATH")

if ld_library_path is not None:
    lib_paths = ld_library_path.split(":")
else:
    lib_paths = []

include_path = os.getenv("INCLUDE_PATH")
if include_path is not None:
    include_paths = [p.strip() for p in include_path.split(":") if len(p.strip()) > 0]
else:
    include_paths = []


# get all the c files that are not cythonized .pyx files.
c_files   = dict( (d, [f for f in glob(join(d, "*.c"))
                       if (f[:-2] + '.pyx') not in all_cython_files])
                  for d in source_directory_list + ['.'])

for d, l in chain(((d, glob(join(d, "*.cxx"))) for d in source_directory_list + ['.']),
                  ((d, glob(join(d, "*.cpp"))) for d in source_directory_list + ['.'])):
    c_files[d] += l


print "C Extension Files Found: \n%s\n+++++++++++++++++++++" % ", ".join(sorted(chain(*c_files.values())))

# Collect all the python modules
def get_python_modules(f):
    d, m = split(f[:f.rfind('.')])
    return m if len(d) == 0 else d + "." + m

exclude_files = set(["setup.py"])
python_files = set(chain(* (list(glob(join(d, "*.py")) for d in source_directory_list) + [glob("*.py")]))) 
python_files -= exclude_files

python_modules = [get_python_modules(f) for f in python_files]

print "Relevant Python Files Found: \n%s\n+++++++++++++++++++++" % ", ".join(sorted(python_files))

if __name__ == '__main__':
    # The rest is also shared with the setup.py file, in addition to
    # this one, so 

    def strip_empty(l):
        return [e.strip() for e in l if len(e.strip()) != 0]

    def get_include_dirs(m):
        return strip_empty(extra_include_dirs + include_paths)

    def get_library_dirs(m):
        return strip_empty(extra_library_dirs + lib_paths)

    def get_libraries(m):
        return strip_empty(library_includes + (specific_libraries[m] if m in specific_libraries else []))
    
    def get_extra_compile_args(m):
        return strip_empty(compiler_args + (['-g', '-O0', '-UNDEBUG']
                                            if debug_mode_c_code
                                            else ['-DNDEBUG']))
    
    def get_extra_link_args(m):
        return strip_empty(link_args + (['-g'] if debug_mode_c_code else []))


    ############################################################
    # Cython extension lists

    def makeExtensionList(d, filelist):
        ext_modules = []

        for f in filelist:
            f_no_ext = f[:f.rfind('.')]
            f_mod = split(f_no_ext)[1]
            modname = "%s.%s" % (d, f_mod) if d != '.' else f_mod
            
            ext_modules.append(Extension(
                    modname,
                    [f],
                    include_dirs = get_include_dirs(modname),
                    library_dirs = get_library_dirs(modname),
                    language = "c++",
                    libraries = get_libraries(modname),
                    extra_compile_args = get_extra_compile_args(modname),
                    extra_link_args = get_extra_link_args(modname),
                    ))

        return ext_modules

    ############################################################
    # Now get all these ready to go

    ext_modules = []

    if cython_mode:
        from Cython.Distutils import build_ext

        ext_modules += list(chain(*list(makeExtensionList(d, l) 
                                        for d, l in cython_files.iteritems())))
        
        cmdclass = {'build_ext' : build_ext}
    else:
        cmdclass = {}

    ext_modules += list(chain(*list(makeExtensionList(d, l)
                                    for d, l in c_files.iteritems())))
    setup(
        version = version,
        description = description,
        author = author, 
        author_email = author_email,
        name = name,
        cmdclass = cmdclass,
        ext_modules = ext_modules,
        py_modules = python_modules,
        scripts = scripts,
        classifiers = classifiers,
        url = url,
        download_url = download_url)
