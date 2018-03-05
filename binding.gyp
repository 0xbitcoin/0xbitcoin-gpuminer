{

"targets":
[
  {
    "target_name": "hybridminer",
    "sources": [
      "cpp/hybridminer/addon.cc",
      "cpp/hybridminer/hybridminer.cpp",
      "cpp/hybridminer/cpusolver.cpp",
      "cpp/hybridminer/gpusolver.cu"
    ],

    'rules': [{
      'extension': ['cpp', 'cc'],
      'cflags_cc+': [ '-march=native', '-O3', '-std=c++17' ],
      "include_dirs": ["<!(node -e \"require('nan')\")"],
      }],

    'rules': [{
     'extension': 'cu',
     'inputs': ['<(RULE_INPUT_PATH)'],
     'outputs':[ '<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o'],
     'rule_name': 'cuda on linux',
     'message': "compile cuda file on linux",
     'process_outputs_as_sources': 1,
     'action': [
        'nvcc', 
        '-std=c++11',
        '-Xcompiler',
        '-fpic',
        '-c',
        '-o',
        '<@(_outputs)',
        '<@(_inputs)'
      ],
    }],

    'variables': {
      'cuda_root%': '$(CUDA_PATH)'
    },

    'libraries': [
      '-lcuda',
      '-lcudart'
    ],

    'library_dirs': [
      '/usr/local/lib',
      '/usr/local/cuda/lib64'
    ],

    "include_dirs": [
      "<(cuda_root)/include",
      "/usr/local/cuda/include",
      "/usr/local/include"
    ]
  }
]
}
