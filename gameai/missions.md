### Creating a new mission in Malmo

Some of the information has been extracted from [this document](https://github.com/microsoft/malmo/blob/master/Schemas/MissionHandlers.xsd)
The official documentation of the XML files can be found [here](https://microsoft.github.io/malmo/0.30.0/Schemas/Mission.html)

## ServerHandlers

To create a chamber we can use a ```ClassRoomDecorator``` and set its specification. See findthegoal.xml for example usage


## AgentHandlers

```AgentStart``` - can be used to set the agent's starting positions and direction. 

```VideoProducer``` - The rendered window size can be set.

```ObservationFromNearbyEntities``` - Malmo returns some numerical information of the entities within a set range. 
```ObservationFromGrid``` - Can use it to construct a top-down symbolic representation.

## Movements

```DiscreteMovementCommands``` - Agent is controlled by a discrete action space.
```ContinousMovementCommands``` - Agent is controlled by a continous action space. 

