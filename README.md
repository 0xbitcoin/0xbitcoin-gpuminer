
# 0xBitcoin ERC20 Token Miner

Solves proof of work to mine supported ERC20 tokens.  



### Update 2.0.11 -GPU & CPU & Pool Mining

This new build adds CUDA GPU Mining capability credits to Mikers! This will mine to a pool or solo and will mine using a CPU or a CUDA GPU.  


IMPORTANT NOTE:  HASHRATE MAY NOT DISPLAY - BUT IT IS WORKING
If unsure, be sure to test functionality with:
'test mine cuda'



Commands :

- mine
- pool mine
- mine cuda *uses nvidia gpu*  
- pool mine cuda *uses nvidia gpu*  
- mine opencl *uses amd gpu*  (not ready)
- pool mine opencl *uses amd gpu* (not ready)
- test mine
- test mine cuda *uses nvidia gpu*  




## 2.0.0 Development Notes  - Adds GPU mining

1. Uses gcc 5  and g++ 5
2. Uses nvcc 9 (nvidia driver and build tools for cuda)
3. A single cpp addon is used to mine with cpu and gpu, it is named hybridminer  
4. The function 'sethardwaretype' switches the mining loop within hybridminer to use a cuda approach or a cpu approach



INSTRUCTIONS : FOLLOW THESE STEPS TO TEST

nvm use 8

npm run build

node index.js

pool mine cuda



### Official Releases Downloads

[Download Packaged Releases](https://github.com/0xbitcoin/0xbitcoin-gpuminer/blob/master/RELEASES.md)



### Building from Source

#### Setup (Windows/Linux)
1. Install NodeJS 8.9
2. Run 'npm install yarn -g' to install yarn package manager
3. Clone/download the project
4. Open a terminal, cd into the project folder and run 'yarn' to install dependencies
5. Run the command 'npm run build' to build C files with node-gyp
6. Start the miner with 'node index.js'

#### Setup (Mac)
1. Install Homebrew & NodeJS 8.9
2. Run 'brew install yarn' to install yarn package manager
3. Clone/download this project
4. Open a terminal, cd into the project folder and run 'yarn'
5. Run the command 'npm run build' to build C files with node-gyp
6. Start the miner with 'node index.js'



### Commands

      {commands}
      "help" - Show the help menu

      "account new" - Create a new mining account
      "account list" - List all mining accounts
      "account select 0x####" - Select a primary mining account by address
      "account balance" - List the ether and token balance of your selected account

      "contract list" - List the selected token contract to mine
      "contract select 0x####" - Select a PoW token contract to mine

      "config gasprice #" - Set the gasprice used to submit PoW to the token smartcontract
      "config cpu_threads #" - Set the number of CPU cores to use for mining
      "config web3provider http://----:####" - Set the web3 provider url for submitting ethereum transactions

      "pool mine" - Begin mining into a pool
      "pool list" - List the selected mining pool
      "pool select http://####.com:####" - Select a pool to mine into

      "mine" - Begin mining solo, directly into the smartcontract




---------------

### Getting Started
1. Build a new mining account with 'account new'
2. View the private key with 'account list'
3. Write down these credentials
4. Mine 0xbitcoin tokens with the command 'mine'

Note that IF SOLO MINING it is necessary to fill the mining account (it is an Ethereum account) with a small amount of ether.  Typically 0.005 eth is good enough to get started.  The ether is used for gas to make function calls to the token smart contract when your miner finds a solution to the Proof of Work.  When the gas is spent that means that you have found a solution! If you were the first to find it, you will be rewarded with 0xbitcoin tokens.  (See the block explorer for typical gas prices at the current moment.)



### Pool Mining
- You can mine into a pool with the command 'pool mine'  
- When mining into a pool, your gasprice does not matter and you will pay NO GAS FEES  
- Every pool is different so consult each pool owner.  Typically, pools will offer a token withdraw mechanism or automatically send tokens to your address on a periodic basis or when a limit is reached



### Vault Datafiles

(requires 'show hidden files and folders')

Stored at:

- Windows
    '/Users/{user}/Appdata/Roaming/.0xbitcoin'

- Mac
    '/home/{user}/Library/Preferences/.0xbitcoin'

- Linux
    '/home/{user}/.0xbitcoin'




### Testing

npm run test


## Credits

1. Zegordo (Developed the accelerated CPU Miner)

        ETH address [0x8AE981d92875C88f713600EB7dC4D23FA7E0E621]



## Tokens that can be mined using Proof of Work:

1. 0xBitcoin token - http://0xbitcoin.org - https://github.com/0xbitcoin/0xbitcoin-token
