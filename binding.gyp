{
'conditions': [
    [ 'OS=="win"', {'variables': {'obj': 'obj'}},
    {'variables': {'obj': 'o'}}]],

"targets": [


 {
      "target_name": "hybridminer",
      "sources": [
        "cpp/hybridminer/addon.cc",
        "cpp/hybridminer/hybridminer.cpp",
        "cpp/hybridminer/cpusolver.cpp",
        "cpp/hybridminer/gpusolver.cpp",
        "cpp/hybridminer/sha3.c"
      ],
      'cflags_cc+': [ '-march=native', '-O3', '-std=c++17' ],
      "include_dirs": ["<!(node -e \"require('nan')\")"],




    }

]
}
