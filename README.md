# Malmö #

Project Malmö is a platform for Artificial Intelligence experimentation and research built on top of Minecraft. We aim to inspire a new generation of research into challenging new problems presented by this unique environment.

[![Join the chat at https://gitter.im/Microsoft/malmo](https://badges.gitter.im/Microsoft/malmo.svg)](https://gitter.im/Microsoft/malmo?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Build Status](https://travis-ci.org/Microsoft/malmo.svg?branch=master)](https://travis-ci.org/Microsoft/malmo) [![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/Microsoft/malmo/blob/master/LICENSE.txt)
----
![malmo](gameai/imgs/malmo_e1_l56.00_r-50.00.gif) 

<!-- <img src="gameai/malmo_e1_l56.00_r-50.00.gif" alt="malmo gif">-->

# Recommended setup method on your own machine
- 1, clone malmo (```git clone https://github.com/martinballa/malmo```)
- 2, change branch to "gameai" : ```git checkout gameai```
- 3, install java 8 and python 3. 
- 4, ```cd malmo/``` and install malmo using pip ```pip install -e MalmoEnv/``` 
- 5, Test if Malmo works correctly by running ```main.py``` in the ```gameai``` folder
- (optional): to run malmo headless on a linux headless server you should install xvfb ```sudo apt-get install -y xvfb```

*Note:* Minecraft uses gradle to build the project and it's not compatible with newer versions of Java, so make sure that you use java version 8 for the build. Make sure that $JAVA_HOME is pointing to the correct version.


# Running on the ITL machines 

```
# clone repo and change branch
git clone https://github.com/martinballa/malmo
git checkout gameai
cd malmo/

module load java/1.8.0_181-oracle
# create python virtualenv
python -m venv <venv> # <venv> is the name of you virtualenv i.e: "malmoenv" or "venv"
source <venv>/bin/activate

# install opencv and malmoenv
pip install -r gameai/requirements.txt
pip install -e MalmoEnv/

cd gameai
python main.py
# this should startup Minecraft and display it on the screen. Note that his might take 1-2 minutes.
```

## Symbolic Observations
The symbolic representations shows a top-down perspective of the environment. There are 2 implementations to extract symbolic representations from Malmo, they are currently specific to the mob_chase_single.xml mission, but can be adapted to any other mission.

### SymbolicObs wrapper
Represent each cell as a single entry on the grid. The entries are mapped to RGB or Grayscale colours, depending on the arguments passed to the wrapper. As each entity (agents, chicken, pig) have direction, these directions are represented by adding a small value to the agent's colour in each direction.
![](gameai/imgs/symbObs.png)

### MultiEntrySymbolicObs wrapper
Represent each cell with 4 values, to better represent the direction and the layers (if the agent is standing on the grass both grass and the agent are visible). Directions are represented by placing the entity in the correct direction on the grid.
 ![](gameai/imgs/multiEntryObs.png)


## Changes from Master
- added examples and merged repo for easier setup - no need to setup both malmo and the example project
- This version has the launcher and some other minor fixes that make working with malmo easier

Each Minecraft instance require a new directory to run it, so using the launcher copies Minecraft into the /tmp directory. In case of failure in /tmp/malmo_<hash>/malmo/out.txt provides the console output from the startup, which can help in debugging. 

## Usage
Instance manager + mission file + arguments

When starting the Malmo instances it might take some time. Note that when using more than one instances ```launch_minecraft``` creates copies of the Minecraft directory in ```/tmp``` as each Minecraft instance requires its own directory. This process can take time.

### Setting JAVA_HOME on Ubuntu this can be done 
```
sudo gedit /etc/profile
```
And add the following lines to the end, where ```<JDK dir>``` is the correct directory for java 8 on your system.
```
JAVA_HOME=/usr/lib/jvm/<JDK dir>
PATH=$PATH:$HOME/bin:$JAVA_HOME/bin
export JAVA_HOME
export JRE_HOME
export PATH
```
Log out and log back in to update the profile settings.

### Installing Java 8 on Mac
If you use multiple java versions, it is recommended to install jenv. [This is a useful post on how to install and use it](https://medium.com/@brunofrascino/working-with-multiple-java-versions-in-macos-9a9c4f15615a)
```
# Download java8 
brew cask install adoptopenjdk/openjdk/adoptopenjdk8

# Install jenv to set java8 as the global version on your Mac
brew install jenv
jenv add /Library/Java/JavaVirtualMachines/<jdk8>/Contents/Home
jenv versions
jenv global 1.8 #(the jdk version pointing to 8)
```

# Original Readme below

## Getting Started ##

### MalmoEnv ###

MalmoEnv implements an Open AI "gym"-like environment in Python without any native code (communicating directly with Java Minecraft). If you only need this functionallity then please see [MalmoEnv](https://github.com/Microsoft/malmo/tree/master/MalmoEnv). This will most likely be the preferred way to develop with Malmo Minecraft going forward.

If you wish to use the "native" Malmo implementation, either install the "Malmo native Python wheel" (if available for your platform) or a pre-built binary release (more on these options below). Building Malmo yourself from source is always an option!

Advantages:
    
1. No native code - you don't have to build or install platform dependent code.
2. A single network connection is used to run missions. No dynamic ports means it's more virtualization friendly.
3. A simpler multi-agent coordination protocol. 
One Minecraft client instance, one single port is used to start missions.
4. Less impedance miss-match with the gym api.

Disadvantages:

1. The existing Malmo examples are not supported (as API used is different). 
Marlo envs should work with this [port](https://github.com/AndKram/marLo/tree/malmoenv).
2. The API is more limited (e.g. selecting video options) - can edit mission xml directly.

### Malmo as a native Python wheel ###

On common Windows, MacOSX and Linux variants it is possible to use ```pip3 install malmo``` to install Malmo as a python with native code package: [Pip install for Malmo](https://github.com/Microsoft/malmo/blob/master/scripts/python-wheel/README.md). Once installed, the malmo Python module can be used to download source and examples and start up Minecraft with the Malmo game mod. 

Alternatively, a pre-built version of Malmo can be installed as follows:

1. [Download the latest *pre-built* version, for Windows, Linux or MacOSX.](https://github.com/Microsoft/malmo/releases)   
      NOTE: This is _not_ the same as downloading a zip of the source from Github. _Doing this **will not work** unless you are planning to build the source code yourself (which is a lengthier process). If you get errors along the lines of "`ImportError: No module named MalmoPython`" it will probably be because you have made this mistake._

2. Install the dependencies for your OS: [Windows](doc/install_windows.md), [Linux](doc/install_linux.md), [MacOSX](doc/install_macosx.md).

3. Launch Minecraft with our Mod installed. Instructions below.

4. Launch one of our sample agents, as Python, C#, C++ or Java. Instructions below.

5. Follow the [Tutorial](https://github.com/Microsoft/malmo/blob/master/Malmo/samples/Python_examples/Tutorial.pdf) 

6. Explore the [Documentation](http://microsoft.github.io/malmo/). This is also available in the readme.html in the release zip.

7. Read the [Blog](http://microsoft.github.io/malmo/blog) for more information.

If you want to build from source then see the build instructions for your OS: [Windows](doc/build_windows.md), [Linux](doc/build_linux.md), [MacOSX](doc/build_macosx.md).

----

## Problems: ##

We're building up a [Troubleshooting](https://github.com/Microsoft/malmo/wiki/Troubleshooting) page of the wiki for frequently encountered situations. If that doesn't work then please ask a question on our [chat page](https://gitter.im/Microsoft/malmo) or open a [new issue](https://github.com/Microsoft/malmo/issues/new).

----

## Launching Minecraft with our Mod: ##

Minecraft needs to create windows and render to them with OpenGL, so the machine you do this from must have a desktop environment.

Go to the folder where you unzipped the release, then:

`cd Minecraft`  
`launchClient` (On Windows)  
`./launchClient.sh` (On Linux or MacOSX)

or, e.g. `launchClient -port 10001` to launch Minecraft on a specific port.

on Linux or MacOSX: `./launchClient.sh -port 10001`

*NB: If you run this from a terminal, the bottom line will say something like "Building 95%" - ignore this - don't wait for 100%! As long as a Minecraft game window has opened and is displaying the main menu, you are good to go.*

By default the Mod chooses port 10000 if available, and will search upwards for a free port if not, up to 11000.
The port chosen is shown in the Mod config page.

To change the port while the Mod is running, use the `portOverride` setting in the Mod config page.

The Mod and the agents use other ports internally, and will find free ones in the range 10000-11000 so if administering
a machine for network use these TCP ports should be open.

----

## Launch an agent: ##

#### Running a Python agent: ####

```
cd Python_Examples
python3 run_mission.py
``` 

#### Running a C++ agent: ####

`cd Cpp_Examples`

To run the pre-built sample:

`run_mission` (on Windows)  
`./run_mission` (on Linux or MacOSX)

To build the sample yourself:

`cmake .`  
`cmake --build .`  
`./run_mission` (on Linux or MacOSX)  
`Debug\run_mission.exe` (on Windows)

#### Running a C# agent: ####

To run the pre-built sample (on Windows):

`cd CSharp_Examples`  
`CSharpExamples_RunMission.exe`

To build the sample yourself, open CSharp_Examples/RunMission.csproj in Visual Studio.

Or from the command-line:

`cd CSharp_Examples`

Then, on Windows:  
```
msbuild RunMission.csproj /p:Platform=x64
bin\x64\Debug\CSharpExamples_RunMission.exe
```

#### Running a Java agent: ####

`cd Java_Examples`  
`java -cp MalmoJavaJar.jar:JavaExamples_run_mission.jar -Djava.library.path=. JavaExamples_run_mission` (on Linux or MacOSX)  
`java -cp MalmoJavaJar.jar;JavaExamples_run_mission.jar -Djava.library.path=. JavaExamples_run_mission` (on Windows)

#### Running an Atari agent: (Linux only) ####

```
cd Python_Examples
python3 ALE_HAC.py
```

----

# Citations #

Please cite Malmo as:

Johnson M., Hofmann K., Hutton T., Bignell D. (2016) [_The Malmo Platform for Artificial Intelligence Experimentation._](http://www.ijcai.org/Proceedings/16/Papers/643.pdf) [Proc. 25th International Joint Conference on Artificial Intelligence](http://www.ijcai.org/Proceedings/2016), Ed. Kambhampati S., p. 4246. AAAI Press, Palo Alto, California USA. https://github.com/Microsoft/malmo

----

# Code of Conduct #

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
