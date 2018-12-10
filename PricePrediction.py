## Load stock prices
import tensorflow as tf
# Note: Once you enable eager execution, it cannot be disabled.
tf.enable_eager_execution()
import numpy as np
import random
import os
import time
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt



def _readFromFile(filename):
    data = []
    file = open(filename, 'r')
    try:
        for line in file:
            line = line[1:-2]
            li = list(line.split(","))
            item = []
            # [date, open, high, low, close, volume, average, barCount]
            # keep date, close and volume
            for i, x in enumerate(li):
                if i == 0:
                    y = x.strip("'")
                    dt = y[:8]
                    tm = y[-8:]
                    item.append(dt)
                    # if there's time following date, separate them and store both
                    if len(x) == 20:
                        item.append(tm)
                elif i == 4:
                    item.append(float(x.strip()))
                elif i == 5:
                    item.append(int(x.strip()))
            data.append(item)
    finally:
        file.close()
    return data


def _prepareData(histPrices, daysPerChunk, stepSize, testRatio):
    # Normalize data: take log2 of the price change ratio
    # and divide by standard deviation
    logPriceChange = [np.log2(histPrices[i + 1] / histPrices[i]) for i in range(len(histPrices) - 1)]
    stdDev = np.std(logPriceChange)
    meanL = np.mean(logPriceChange)
    sdLogPriceChange = [(x - meanL) / stdDev for x in logPriceChange]

    inputs = []
    targets = []

    for f in range(0, len(sdLogPriceChange) - daysPerChunk, stepSize):
        input = sdLogPriceChange[f: f+daysPerChunk]
        target = sdLogPriceChange[f+1: f+daysPerChunk+1]
        inputs.append(input)
        targets.append(target)

    combined = list(zip(inputs, targets))
    random.shuffle(combined)
    inputs[:], targets[:] = zip(*combined)

    train_size = int(len(inputs) * (1.0 - testRatio))
    train_X, test_X = inputs[:train_size], inputs[train_size:]
    train_Y, test_Y = targets[:train_size], targets[train_size:]
    return train_X, train_Y, test_X, test_Y, stdDev, meanL


dailyData = _readFromFile("data/SPY_1 day.txt")
minuteData = _readFromFile("data/SPY_3 mins.txt")

dailyDates = [datetime.strptime(item[0], '%Y%m%d') for item in dailyData]
dailyClosePrices = [item[1] for item in dailyData]
dailyVolumes = [item[2] for item in dailyData]
minuteDateTime = [datetime.combine(datetime.strptime(item[0], '%Y%m%d'),
                                   datetime.strptime(item[1], '%H:%M:%S').time()) for item in minuteData]
minuteClosePrices = [item[2] for item in minuteData]
minuteVolumes = [item[3] for item in minuteData]

# setting the length of days we want for a single input
daysPerChunk = 20
stepSize = 1
testRatio = 0.2
outputDim = 1
# number of GRU RNN units
units = 1024
# batch size
BATCH_SIZE = 64
# buffer size to shuffle our dataset
BUFFER_SIZE = 10000

train_X, train_Y, test_X, test_Y, stdDev, meanL = _prepareData(dailyClosePrices[::10], daysPerChunk, stepSize, testRatio)
# train_X, train_Y, test_X, test_Y, stdDev, meanL = _prepareData(minuteClosePrices, daysPerChunk, stepSize, testRatio)

# print (np.array(train_X).shape)
# print (np.array(train_Y).shape)
# print (np.array(test_X).shape)
# print (np.array(test_Y).shape)

trainDataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).shuffle(BUFFER_SIZE)
trainDataset = trainDataset.batch(BATCH_SIZE, drop_remainder=True)


##
class Model(tf.keras.Model):
    def __init__(self, units, outputDim, batch_size):
        super(Model, self).__init__()
        self.units = units
        self.batch_sz = batch_size

        # self.embedding = tf.keras.layers.Embedding(outputDim, embeddingDim)
        self.gru = tf.keras.layers.CuDNNGRU(self.units,
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(outputDim)

    def call(self, x, hidden):
        # x = self.embedding(x)

        # output shape == (batch_size, daysPerChunk, hidden_size)
        # states shape == (batch_size, hidden_size)

        # hidden_states variable to preserve the state of the model
        # this will be used to pass at every step to the model while training

        # print("hidden before GRU:")
        # print(hidden)
        # print("x before GRU:")
        # print(x)
        output, hidden_states = self.gru(x, initial_state=hidden)
        # print("hidden after GRU:")
        # print(hidden)
        # print("x after GRU:")
        # print(output)

        # print("output from GRU:")
        # print(output.shape)

        # reshaping the output so that we can pass it to the Dense layer
        # after reshaping the shape is (batch_size * daysPerChunk, hidden_size)
        # output = tf.reshape(output, (-1, output.shape[2]))

        # only take output of the last step from RNN
        output = output[:,-1,:]

        # The dense layer will output predictions for every time_steps(daysPerChunk)
        # output shape after the dense layer == (daysPerChunk * batch_size, outputDim)
        x = self.fc(output)
        # print("x of final dense layer:")
        # print(x.shape)

        return x, hidden_states


model = Model(units, outputDim, BATCH_SIZE)
optimizer = tf.train.AdamOptimizer()


# using sparse_softmax_cross_entropy so that we don't have to create one-hot vectors
def loss_func(real, preds):
    # return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)
    return tf.losses.mean_pairwise_squared_error(labels=real, predictions=preds)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=model)

# Training step
EPOCHS = 20

for epoch in range(EPOCHS):
    start = time.time()

    # initializing the hidden state at the start of every epoch
    hidden = model.reset_states()

    for (batch, (inp, target)) in enumerate(trainDataset):
        with tf.GradientTape() as tape:
            # feeding the hidden state back into the model
            # This is the interesting step

            # print("input shape before reshaping:")
            # print(inp.shape)
            inp = tf.reshape(inp, (inp.shape[0], inp.shape[1], 1))
            # print("input shape after reshaping:")
            # print(inp.shape)

            predictions, hidden = model(inp, hidden)
            # print("hidden:")
            # print(hidden.shape)

            # reshaping the target because that's how the
            # loss function expects it
            # print(target.shape)
            # print(predictions.shape)
            # target = tf.reshape(target, (-1,1))
            target = tf.expand_dims(target[:,-1], 1)

            # print("target shape:")
            # print(target.shape)
            loss = loss_func(target, predictions)

        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, loss))

    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

##
# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# empty string to store our results
priceGenerated = []
priceReal = [x[-1] for x in test_Y]
# print(np.array(test_Y))
# print(np.array(priceReal))

for i in range(len(test_X)):
    inputVal = test_X[i]
    inputVal = tf.expand_dims(inputVal, 0)
    inputVal = tf.expand_dims(inputVal, 2)

    # hidden state shape == (batch_size, number of rnn units); here batch size == 1
    hidden = tf.zeros((1, units))
    hidden = tf.cast(hidden, tf.float64)

    predicted_val, hidden = model(inputVal, hidden)
    # print(predicted_val)
    # print(predicted_val[-1,0])
    # print(predicted_val[-1])

    priceGenerated.append(predicted_val[-1,0])
    # print(priceGenerated)



# plt.plot(priceGenerated)
# plt.gcf().autofmt_xdate()
# plt.show()



## Result
from scipy import stats

# print(np.mean(priceGenerated))
# print(np.mean(priceReal))

sameSignList = [x*y>0 for x,y in zip(priceGenerated, priceReal)]
accuracy = tf.reduce_sum(tf.cast(sameSignList, tf.float32))/len(sameSignList)
# print(accuracy)

diff_original = sum([np.square(x) for x in priceReal])
diff = sum([np.square(x-y) for x,y in zip(priceGenerated, priceReal)])
print("Original Loss:")
print(diff_original)
print("Prediction Loss:")
print(diff)

lowerPercentile = 20
pLower = np.percentile(priceGenerated, lowerPercentile)
pHigher = np.percentile(priceGenerated, 100-lowerPercentile)

countTrue = 0
count = 0
countLTrue = 0
countL = 0
countHTrue = 0
countH = 0

for i in range(len(sameSignList)):
    if priceGenerated[i] <= pLower or priceGenerated[i] >= pHigher:
        if sameSignList[i]:
            countTrue += 1
        count += 1

    if priceGenerated[i] >= pHigher:
        if sameSignList[i]:
            countHTrue += 1
        countH += 1

    if priceGenerated[i] <= pLower :
        if sameSignList[i]:
            countLTrue += 1
        countL += 1
print(countLTrue, countL)
print(countLTrue/countL)
print(countHTrue, countH)
print(countHTrue/countH)
print(countTrue, count)
print(countTrue/count)
print(accuracy)

