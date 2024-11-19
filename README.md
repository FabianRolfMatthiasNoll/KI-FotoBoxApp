# Instalation

## Setting up Ollama and LLAVA

First install Ollama on your linux system. In my case  
`sudo pacman -S ollama`

Then the ollama service needs to be started  
`ollama serve`

After that the model can be pulled. In the case of this application LLAVA 34b is currently used.  
`ollama pull llava:34b`

## Setting up U2Net

For using U2Net (currently used for background segmentation) the u2net.pth file is needed in the directory `./u2net/u2net.pth`

This file can be downloaded from https://github.com/xuebinqin/U-2-Net. There is a file download for the pretrained model with ~173MB

