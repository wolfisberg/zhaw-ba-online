# Audio file down sampling

## Dependencies

### Via Poetry (porject dependency manager)
    sudo pacman -S poetry
    # in the project root, where to .toml file is
    poetry install

### Manually
- librosa
- soundfile
- numpy

## Version mismatches
Surfaced for me because ```poetry add librosas``` failed to install the ```llvmlite``` dependency. This was because llvmlite is not compatible with Python3.9 (yet).

Python versions can be managed via ```pyenv```.

    sudo pacman -S pyenv
    pyenv install 3.8.7
    poetry env use /home/<name>/.pyenv/versions/3.8.7/bin/python

    # then
    poetry add librosa
    # or if it is already in your .toml dependency file
    poetry install

## Usage

    python downsampler.py <target dir>

Reduces quality of audio files ```['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav']``` found in ```<target dir>``` to 16kHz sample rate and 16bit depth ([librosa doc](https://librosa.org/doc/latest/generated/librosa.util.find_files.html)).
