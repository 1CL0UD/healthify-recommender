import math
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import DenseFeatures
from tensorflow.feature_column import categorical_column_with_hash_bucket, numeric_column, embedding_column

data = pd.read_excel("dataset/USDA-TKPI_Food_Dataset (1).xls")

shuffled = data.sample(frac=1).reset_index(drop=True)

def calculate_caloric_intake(weight, weight_target, pace, Macro):
    if pace == 'normal':
        daily = 500  # 0.5 kg per week
        weekly = 3500
    elif pace == 'strict':
        daily = 1000  # 1.0 kg per week
        weekly = 7000
    else:
        print("Invalid pace input. Please enter 'normal' or 'strict'.")
        return

    change = weight - weight_target
    calorie_need = change * weekly
    if change < 0:
      if pace == 'strict':
        calorie = Macro + daily
        weeks = abs(change)
        print(f'calorie daily per week for surplus = {calorie} calorie, for {weeks} weeks')
        return calorie, weeks

      elif pace == 'normal':
        calorie = Macro + daily
        weeks = abs(change) * 2
        print(f'calorie daily per week for surplus = {calorie} calorie, for {weeks} weeks')
        return calorie, weeks

    if change > 0 :

      if pace == 'strict':
        calorie = Macro - daily
        weeks = abs(change)
        print(f'calorie daily per week for deficit = {calorie} calorie, for {weeks} weeks')
        return calorie, weeks

      elif pace == 'normal':
        calorie = Macro - daily
        weeks = abs(change) * 2
        print(f'calorie daily per week for deficit = {calorie} calorie, for {weeks} weeks')
        return calorie, weeks



sex = 'male'
age = 20
weight = 70
height = 176
activity_level = 1.4
weight_target = 75
pace = 'normal'
vegetarian = 'yes'


def BMRf(sex, age, weight, height):
  if sex == 'male':
    BMR = 10 * weight + 6.25 * height - 5 * age  + 5
    return BMR
  else:
    BMR = 10 * weight + 6.25 * height - 5 * age  - 161
    return BMR

def macrof(BMR, activity_level):
  macro = BMR * activity_level
  return macro

def recommendation_function(sex, age, weight, height, activity_level, weight_target, pace, vegetarian):
  BMR = BMRf(sex, age, weight, height)

  Macro = macrof(BMR, activity_level)

  Calorie, weeks = calculate_caloric_intake(weight, weight_target, pace, Macro)



  # Number of items
  num_items = len(shuffled)

  # Weights and values
  weights = shuffled['Lemak(g)'].values  # We use fat as the weight
  values = shuffled['Energi(Kal)'].values  # We use calories as the value

  # Shuffle the data
  indices = np.arange(num_items)
  np.random.shuffle(indices)

  weights = weights[indices]
  values = values[indices]

  # Knapsack capacity
  capacity = 22  # This is the total amount of fat we want to minimize

  # State: (current weight, item, total value)
  state = tf.keras.Input(shape=(3,))

  # Q-values: (do nothing, pick item)
  q_values = tf.keras.layers.Dense(2)(state)

  # Model
  model3 = tf.keras.Model(inputs=state, outputs=q_values)

  # Loss and optimizer
  loss_fn = tf.keras.losses.MeanSquaredError()
  optimizer = tf.keras.optimizers.Adam()

  # Discount factor
  gamma = 0.95

  # Epsilon for epsilon-greedy strategy
  epsilon = 1.0
  epsilon_decay = 0.995

  # Training loop
  stop_training = False
  selected_foods = []  # Initialize an empty list to store selected foods

  for episode in range(1000):  # Increase the number of episodes
      # Initial state
      current_weight = 0
      total_value = 0

      for item in range(num_items):
          # Current state
          current_state = np.array([[current_weight, item, total_value]])

          # Compute Q-values
          current_q_values = model3.predict(current_state)

          # Choose action
          if np.random.rand() < epsilon:
              action = np.random.randint(2)
          else:
              action = np.argmax(current_q_values)

          # Update current weight and total value
          if action == 1 and current_weight + weights[item] <= capacity:
              current_weight += weights[item]
              total_value += values[item]

              # Save the selected item if the weight and value constraints are met
              selected_foods.append((shuffled['Nama Bahan Makanan'][item], values[item]))

          # Print the weights and total value per loop
          print(f"Current weight: {current_weight}, Total value: {total_value}")

          # Compute reward
          reward = total_value - 16 * current_weight  # Adjust the reward function

          # Next state
          next_state = np.array([[current_weight, item + 1, total_value]])

          # Compute next Q-values
          next_q_values = model3.predict(next_state)

          # Compute target Q-values
          target_q_values = current_q_values
          target_q_values[0, action] = reward + gamma * np.max(next_q_values)

          # Update model
          with tf.GradientTape() as tape:
              q_values = model3(current_state)
              loss = loss_fn(target_q_values, q_values)
          grads = tape.gradient(loss, model3.trainable_weights)
          optimizer.apply_gradients(zip(grads, model3.trainable_weights))

          # Check if the total value exceeds Calorie and break both loops
          if total_value >= Calorie:
              print(f"Total value exceeds {Calorie}. Stopping training.")
              stop_training = True
              break

      # Decrease epsilon
      epsilon *= epsilon_decay

      if stop_training:
          break
  return selected_foods


