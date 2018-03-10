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
        "cpp/hybridminer/sha3.c",
        "cpp/hybridminer/cudasolver.cu"
      ],
      'conditions': [
		  [ 'OS=="win"',
			{'cflags_cc+': [ '-EHsc', '-W3', '-nologo', '-Ox', '-FS',
                      '-Zi', '-MT', '-I', 'cpp/hybridminer' ],
			},
			{'cflags_cc+': [ '-march=native', '-O3', '-std=c++11' ],
			}
		  ]
	  ],
      "include_dirs": ["<!(node -e \"require('nan')\")"],

      'rules': [
        {
          'extension': 'cu',
          'inputs': ['<(RULE_INPUT_PATH)'],
          'outputs':['<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o'],
          'conditions': [
            [ 'OS=="win"',
              {'rule_name': 'cuda on windows',
               'message': "compile cuda file on windows",
               'process_outputs_as_sources': 0,
               'action': ['nvcc -cudart static --machine 64\
                          -c <(_inputs) -o <(_outputs)',
                          '-gencode=arch=compute_61,code=compute_61',
                          '-gencode=arch=compute_52,code=sm_52',
                          '-gencode=arch=compute_35,code=sm_35',
                          '-I', 'cpp/hybridminer'],
              }, 
              {'rule_name': 'cuda on linux',
               'message': "compile cuda file on linux",
               'process_outputs_as_sources': 1,
               'action': ['nvcc','-std=c++11','-Xcompiler','-fpic','-c',
                          '<@(_inputs)','-o','<@(_outputs)'],
              }
            ]
		  ]
		}],

      'conditions': [
        [ 'OS=="mac"', {
          'libraries': ['-framework CUDA'],
          'include_dirs': ['/usr/local/include'],
          'library_dirs': ['/usr/local/lib'],
        }],
        [ 'OS=="linux"', {
          'libraries': ['-lcuda', '-lcudart'],
          'include_dirs': ['/usr/local/include'],
          'library_dirs': ['/usr/local/lib',
                           '/usr/local/cuda/lib64',
                           './cuda']
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
          'cflags_cc+': [ '-EHsc', '-W3', '-nologo', '-Ox',
                          '-FS', '-Zi', '-MT' ],
          'variables': {
            'cuda_root%': '$(CUDA_PATH)'
          },

          'libraries': [
            'cuda.lib',
            'cudart.lib',
			'cudasolver.o'
          ],

          'library_dirs': [
            '<(cuda_root)/lib/<(arch)',
			'<(module_root_dir)/build/Release/obj/hybridminer'
          ],

          "include_dirs": [
            "<(cuda_root)/include",
			'cpp/hybridminer'
          ]
        }
		]
      ]
    }
  ]
}
