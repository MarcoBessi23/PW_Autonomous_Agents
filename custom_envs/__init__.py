'''
this file registers the custom environments to be used with Gymnasium
'''

from gymnasium.envs.registration import register

register(
    id='MovingObstaclesGrid-v3',
    entry_point='custom_envs.environment:DynamicObstacleShapes',
    max_episode_steps=500
)

register(
    id='MovingObstaclesGrid-v1',
    entry_point='custom_envs.environment:DynamicObstacleShapesSingleChannel',
    max_episode_steps=500
)
