"""
Different task implementations that can be defined inside an ai2thor environment
"""
import math
from gym_ai2thor.utils import InvalidTaskParams
import json


class BaseTask(object):
    """
    Base class for other tasks to subclass and create specific reward and reset functions
    """
    def __init__(self, config):
        self.task_config = config
        self.max_episode_length = config.get('max_episode_length', 1500)
        # default reward is negative to encourage the agent to move more
        self.movement_reward = config.get('movement_reward', -0.01)
        # self.object_num = 0
        self.step_num = 0

    # def set_object_number(self, num):
    #     self.object_num = num

    def transition_reward(self, state):
        """
        Returns the reward given the corresponding information (state, dictionary with objects
        collected, distance to goal, etc.) depending on the task.
        :return: (args, kwargs) First elemnt represents the reward obtained at the step
                                Second element represents if episode finished at this step
        """
        raise NotImplementedError

    def reset(self):
        """

        :param args, kwargs: Configuration for task initialization
        :return:
        """
        raise NotImplementedError


class PickUpTask(BaseTask):
    """
    This task consists of picking up a target object. Rewards are only collected if the right
    object was added to the inventory with the action PickUp (See gym_ai2thor.envs.ai2thor_env for
    details). Because the agent can only carry 1 object at a time in its inventory, to receive
    a lot of reward one must learn to put objects down. Optimal behaviour will lead to the agent
    spamming PickupObject and PutObject near a receptacle. target_objects is a dict which contains
    the target objects which the agent gets reward for picking up and the amount of reward was the
    value
    """
    def __init__(self, **kwargs):
        super(PickUpTask, self).__init__(kwargs)
        # check that target objects are not selected as NON pickupables
        missing_objects = []
        for obj in kwargs['task']['target_objects'].keys():
            if obj not in kwargs['pickup_objects']:
                missing_objects.append(obj)
        if missing_objects:
            raise InvalidTaskParams('Error initializing PickUpTask. The objects {} are not '
                                    'pickupable!'.format(missing_objects))

        self.target_objects = kwargs['task'].get('target_objects', {'Mug': 1})
        self.prev_inventory = []

    def transition_reward(self, state):
        reward, done = self.movement_reward, False
        curr_inventory = state.metadata['inventoryObjects']
        object_picked_up = not self.prev_inventory and curr_inventory and \
                           curr_inventory[0]['objectType'] in self.target_objects

        if object_picked_up:
            # One of the Target objects has been picked up. Add reward from the specific object
            reward += self.target_objects.get(curr_inventory[0]['objectType'], 0)
            print('{} reward collected!'.format(reward))

        if self.max_episode_length and self.step_num >= self.max_episode_length:
            print('Reached maximum episode length: {}'.format(self.step_num))
            done = True

        self.prev_inventory = state.metadata['inventoryObjects']
        return reward, done

    def reset(self):
        self.prev_inventory = []
        self.step_num = 0


class MoveAheadTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super(MoveAheadTask, self).__init__(kwargs)
        self.rewards = []

    def transition_reward(self, state):
        reward = 1 if state.metadata['lastAction'] == 'MoveAhead' else -1
        self.rewards.append(reward)
        done = sum(self.rewards) > 100 or self.step_num > self.max_episode_length
        if done:
            self.rewards = []
        return reward, done

    def reset(self):
        self.step_num = 0


class ObjectFindTaskOff(BaseTask):
    def __init__(self, *args, **kwargs):
        super(ObjectFindTaskOff, self).__init__(kwargs)
        metadata_file = 'metadata.json'
        with open(metadata_file, 'r', encoding='utf-8') as file2:
            self.metadata = json.load(file2)

    def transition_reward(self, ev, map):
        reward = 0
        FindNew = False
        for o in self.metadata[ev]['objects']:
            # if o['visible'] and not map.has_key(o['objectId']):
            if o['visible'] and not o['objectId'] in map:
                reward += 1
                FindNew = True
                position = []
                position.append(o['position']['x'])
                position.append(o['position']['y'])
                position.append(o['position']['z'])
                map[o['objectId']] = position
        if not FindNew:
            reward = -0.01

        done = self.step_num > self.max_episode_length  or len(map) > len(self.metadata[ev]['objects'])

        return reward, done

    def reset(self):
        self.step_num = 0


class ObjectFindTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super(ObjectFindTask, self).__init__(kwargs)

    def transition_reward(self, event, map, collided):
        reward = 0
        FindNew = False
        #visible_objects = [obj for obj in event.metadata['objects'] if obj['visible']]
        #for obj in visible_objects:
        for o in event.metadata['objects']:
            # if o['visible'] and not map.has_key(o['objectId']):
            if o['visible'] and not o['objectId'] in map:
                reward += 3
                FindNew = True
                position = []
                position.append(o['position']['x'])
                position.append(o['position']['y'])
                position.append(o['position']['z'])
                map[o['objectId']] = position
        #print('Find objects num: ' + str(len(map)))

        if not FindNew:
            reward = -0.01

        if collided:
            reward = -20

        #print('step num: ' + str(self.step_num))

        done = self.step_num > self.max_episode_length  or len(map) >= len(event.metadata['objects'])
        #done = len(map) >= len(event.metadata['objects'])

        return reward, done, len(map)

    def reset(self):
        self.step_num = 0


class MoveToObjectTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super(MoveToObjectTask, self).__init__(kwargs)
        self.target_name = kwargs['task'].get('position_of_target')
        self.target_position = [0, 0]
        self.target_object_index = 0
        self.last_dist = 0
        self.curr_dist = 0
        self.last_x = 0
        self.curr_x = 0
        self.last_z = 0
        self.curr_z = 0
        self.negative_rewardx = 0
        self.negative_rewardz = 0
        self.negative_numx = 0
        self.negative_numz = 0


    def transition_reward(self,  event, map, collided):
        done = False
        if self.target_position == [0, 0]:
            for o in event.metadata['objects']:
                if o['objectId'][:o['objectId'].index('|')] == self.target_name:
                    self.target_position = [o['position']['x'], o['position']['z']]
                    self.target_object_index = event.metadata['objects'].index(o)
                    self.last_dist = o['distance']
                    self.last_x = abs(event.metadata['agent']['position']['x'] - o['position']['x'])
                    self.last_z = abs(event.metadata['agent']['position']['z'] - o['position']['z'])
                    print('initial position successfull!')
                    break

        #a_x = event.metadata['agent']['position']['x']
        #a_z = event.metadata['agent']['position']['z']
        #self.curr_dist = math.sqrt(math.pow((a_x - self.target_position[0]), 2) + math.pow((a_z - self.target_position[1]), 2))
        obj = event.metadata['objects'][self.target_object_index]
        self.curr_dist = obj['distance']
        self.curr_x = abs(event.metadata['agent']['position']['x'] -obj['position']['x'])
        self.curr_z = abs(event.metadata['agent']['position']['z'] -obj['position']['z'])
        diff = self.last_dist - self.curr_dist
        diff_x = self.last_x - self.curr_x
        diff_z = self.last_z - self.curr_z

        reward = 0
        curr_p = []
        curr_p.append(event.metadata['agent']['position']['x'])
        curr_p.append(event.metadata['agent']['position']['z'])
        curr_p.append(event.metadata['agent']['rotation']['x'])
        curr_p.append(event.metadata['agent']['rotation']['y'])
        curr_p.append(event.metadata['agent']['rotation']['z'])

        #Rg
        if obj['distance'] <= 1.0 and obj['visible'] == True:
            done = True
            Rg = 1
            print('arrived!')
        else:
            Rg = 0.02*diff - 0.001

        #Rc
        if collided:
            Rc = -0.008
        else:
            Rc = 0

        #Ra
        if curr_p not in map.values():
            map[self.step_num] = curr_p
            if diff > 0:
                Ra = 0.005
            elif diff < 0:
                Ra = 0.0025
            elif diff == 0:
                Ra = 0
        else:
            Ra = -0.01

        reward = Rg + Rc + Ra
        """
        if obj['distance'] <= 1.0 and obj['visible'] == True:
        #if obj['distance'] <= 1.0:
            done = True
            reward = 1
            print('arrived!')
        #if diff > 0 or diff_x > 0 or diff_z > 0:
        if diff > 0:
            #print('diff', diff, ',diff_x', diff_x, ',diff_z', diff_z)
            reward = 0.003
            #reward = 0.003
            # print('better!')
        elif diff < 0:
            reward = -0.004
            #reward = -0.003
            # print('worse!')
        elif diff == 0:
            reward = -0.002
            # print('time cost and collided')
        if collided:
            reward = -0.003
            # reward = -0.002
            # print('collided!')

        if reward < 0 or self.negative_reward < 0:
            self.negative_reward += reward
            if reward > 0:
                self.negative_reward += 8*reward
            if self.negative_reward > 0:
                self.negative_reward = 0
            reward = 0
        """
        """
        #if 往前走步数达到一定时,执行以下两个case
        #case 1
        if diff_x == 0 and diff_z != 0 and reward < 0:
            self.negative_rewardx += reward
            self.negative_numx += 1
            if self.negative_numx < 5:
                reward += self.negative_numx*0.0008
            else:
                reward = 0
        if reward > 0 and diff_x > 0 and self.negative_rewardx < 0:
            self.negative_rewardx = 0
            self.negative_numx = 0
            reward = 2*reward

        #case 2
        if diff_x != 0 and diff_z == 0 and reward < 0:
            self.negative_rewardz += reward
            self.negative_numz += 1
            if self.negative_numz < 5:
                reward += self.negative_numz*0.0008
            else:
                reward = 0
        if reward > 0 and diff_z > 0 and self.negative_rewardz < 0:
            self.negative_rewardz = 0
            self.negative_numz = 0
            reward = 2*reward
        """

        #if done:
        #    reward = 1
        #else:
        #    reward = 0.01 * (self.last_dist - self.curr_dist)

        if self.step_num > self.max_episode_length:
            done = True

        self.last_dist = self.curr_dist
        self.last_x= self.curr_x
        self.last_z = self.curr_z

        return reward, done, 0

    def reset(self):
        self.step_num = 0


class InteractionTask(BaseTask):
        def __init__(self, *args, **kwargs):
            super(InteractionTask, self).__init__(kwargs)
            self.target_object_index = 0
            self.target_objects = None
            self.target_name = kwargs['task'].get('position_of_target')
            self.target_position = [0, 0]
            self.last_dist = 0
            self.curr_dist = 0

        def transition_reward(self, event, map, collided):
            if self.target_position == [0, 0]:
                for o in event.metadata['objects']:
                    if o['objectId'][:o['objectId'].index('|')] == self.target_name:
                        self.target_position = [o['position']['x'], o['position']['z']]
                        self.target_objects = o['objectId']
                        self.target_object_index = event.metadata['objects'].index(o)
                        self.last_dist = o['distance']
                        print('initial object successfull!')
                        break

            #a_x = event.metadata['agent']['position']['x']
            #a_z = event.metadata['agent']['position']['z']
            #self.curr_dist = math.sqrt(math.pow((a_x - self.target_position[0]), 2) + math.pow((a_z - self.target_position[1]), 2))
            obj = event.metadata['objects'][self.target_object_index]
            self.curr_dist = obj['distance']
            diff = self.last_dist - self.curr_dist

            reward = 0
            done = False
            curr_inventory = event.metadata['inventoryObjects']
            object_picked_up = curr_inventory and curr_inventory[0]['objectId'] in self.target_objects

            if object_picked_up:
                done = True
                print ('got it!')

            if diff > 0:
                reward = 0.02
            else:
                reward = -0.03
            if collided:
                reward = -0.01
            if done:
                reward = 3

            if self.step_num > self.max_episode_length:
                done = True

            self.last_dist = self.curr_dist
            return reward, done, len(map)

        def reset(self):
            self.step_num = 0