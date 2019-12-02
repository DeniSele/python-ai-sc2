# python-ai-sc2
Python AI in StarCraft II using genetic algorithm.

## Installation

By installing this library(and maps later) you agree to be bound by the terms of the [AI and Machine Learning License](http://blzdistsc2-a.akamaihd.net/AI_AND_MACHINE_LEARNING_LICENSE.html).

First of all, you need an StarCraft II executable. If you are running Windows or macOS, just install the normal SC2 from blizzard app. Linux users get the best experience by installing the Windows version of StarCraft II with Wine. Linux user can also use the Linux binary, but it's headless so you cannot actually see the game.

Second, you need to install a special API to interact with the game.

You'll need Python 3.6 or newer.

```
pip3 install --user --upgrade sc2
```

Third, you need to download some maps. Official map downloads are available from [Blizzard/s2client-proto](https://github.com/Blizzard/s2client-proto#downloads). Notice: the map files are to be extracted into *subdirectories* of the `StarCraft II/Maps` directory (if you don't have a `Maps` folder, you need to create one). We need the `Ladder 2017 Season 1`. 
The final path may look like this, for example: `D:\StarCraft II\Maps\Ladder2017Season1`.
The files are password protected with the password 'iagreetotheeula'.

And finally, we need to go to sc2 installation folder, find the `paths.py` file there and change the way the game is installed on your computer.

### Running

After installing required librarys, a StarCraft II executable, and some maps, you're ready to get started. Simply run a `training_launcher.py` file. 

Notice: when running each generation, half of its individuals are run simultaneously (if population 8, then 4 instances of Starcraft are run simultaneously). Each one takes up about 1.2 GB of RAM (don't forget to set all settings to a minimum)
