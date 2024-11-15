

# bellow are some examples of how to load data from openx dataset using tfds


# for each dataset you query, you need to look at the respective tfds featuresdict on https://www.tensorflow.org/datasets/catalog/


# example berkeley_ur5_dataset.py

import tensorflow_datasets as tfds
import tensorflow as tf

# Load the dataset builder from its directory
dataset_name = 'berkeley_autolab_ur5'  # Replace with your dataset name
builder = tfds.builder_from_directory(builder_dir=f'gs://gresearch/robotics/{dataset_name}/0.1.0/')

# Create the dataset, specifying to load only the first episode
# usually don't pick the first episode as it it doesn't involve the gripper getting manipulated
ds = builder.as_dataset(split='train[40:41]')  # Load only the first episode

# Define a function to extract robot state from the steps
def extract_robot_state(episode):
    return episode['steps'].map(lambda step: step['observation']['robot_state'])

# Apply the extraction function
robot_states = ds.map(extract_robot_state, num_parallel_calls=tf.data.AUTOTUNE)


# now you can iterate through as you wish
for episode in robot_states:
    for robot_state in episode:
        print(robot_state.numpy().shape)


# below is a similar example for loadiing an xarm dataset but this time extracting two features at once from the dataset: the joint angles and the open/closed gripper state



dataset_name = 'utokyo_xarm_pick_and_place_converted_externally_to_rlds'  # Replace with your dataset name
builder = tfds.builder_from_directory(builder_dir=f'gs://gresearch/robotics/{dataset_name}/0.1.0/')


ds = builder.as_dataset(split='train[40:41]')  


def extract_robot_state_and_gripper(episode):
    def step_map_fn(step):
        return {
            'robot_state': step['observation']['joint_state'],
            'gripper_state': step['action'][-1]  # Extract the last element from the action tensor
        }
    return episode['steps'].map(step_map_fn)


processed_data = ds.map(extract_robot_state_and_gripper, num_parallel_calls=tf.data.AUTOTUNE)


# iterate as you wish
for episode in processed_data:
    for step in episode:
        print("Robot State:", step['robot_state'].numpy())
        print("Gripper State:",step['gripper_state'].numpy() ) # Gripper open/close state


