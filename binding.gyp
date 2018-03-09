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
        "cpp/hybridminer/cudasolver.cu",
        "cpp/hybridminer/sha3.c"

      ],
      'cflags_cc+': [ '-march=native', '-O3', '-std=c++11' ],
      "include_dirs": ["<!(node -e \"require('nan')\")"],



       'rules': [


         {
           'extension': 'cu',
           'inputs': ['<(RULE_INPUT_PATH)'],
           'outputs':[ '<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o'],
           'conditions': [
           [ 'OS=="win"',
              {'rule_name': 'cuda on windows',
               'message': "compile cuda file on windows",
               'process_outputs_as_sources': 0,
               'action': ['nvcc --std=c++11 -c <(_inputs) -o  <(_outputs)'],
               },
             {'rule_name': 'cuda on linux',
               'message': "compile cuda file on linux",
               'process_outputs_as_sources': 1,
               'action': ['nvcc','-std=c++11','-Xcompiler','-fpic','-c',
                  '<@(_inputs)','-o','<@(_outputs)'],
             }
           ]]}],

         'conditions': [
          [ 'OS=="mac"', {
            'libraries': ['-framework CUDA'],
            'include_dirs': ['/usr/local/include'],
            'library_dirs': ['/usr/local/lib'],
          }],
          [ 'OS=="linux"', {
            'libraries': ['-lcuda', '-lcudart'],
            'include_dirs': ['/usr/local/include'],
            'library_dirs': ['/usr/local/lib', '/usr/local/cuda/lib64', './cuda']
          }],
          [ 'OS=="win"', {
            'conditions': [
              ['target_arch=="x64"',
                {
                  'variables': { 'arch': 'x64' }
                }, {
                  'variables': { 'arch': 'Win32' }
                }
              ],
            ],
            'variables': {
              'cuda_root%': '$(CUDA_PATH)'
            },

            'libraries': [
              '-lcuda',
              '-lcudart',
              '<(module_root_dir)/build/Release/obj/hybridminer/cudasolver.o'
            ],

            'library_dirs': [
              '<(cuda_root)/lib/<(arch)',
              '/usr/local/lib',
              '/usr/local/cuda/lib64'
            ],

            "include_dirs": [
              "<(cuda_root)/include",
              "/usr/local/cuda/include",
              "/usr/local/include"
            ]
          }]
        ]
    }

]
}
