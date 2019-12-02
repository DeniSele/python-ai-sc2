import numpy


def sigmoid(inpt):
    return 1.0/(1.0+numpy.exp(-1*inpt))


def relu(inpt):
    result = inpt
    result[inpt < 0] = 0
    return result


def predict_solo_output(weights_mat, data_inputs, activation="relu"):
    # Prediction of the result based on the data provided

    r1 = data_inputs
    for curr_weights in weights_mat:
        r1 = numpy.matmul(r1, curr_weights)
        if activation == "relu":
            r1 = relu(r1)
        elif activation == "sigmoid":
            r1 = sigmoid(r1)
    predicted_label = numpy.where(r1 == numpy.max(r1))[0][0]
    return predicted_label


def predict_outputs(weights_mat, data_inputs, data_outputs, activation="relu"):
    # Predict the results for the population, 
    # calculating the accuracy of the results against the expected results

    predictions = numpy.zeros(shape=(data_inputs.shape[0]))
    for sample_idx in range(data_inputs.shape[0]):
        r1 = data_inputs[sample_idx, :]
        for curr_weights in weights_mat:
            r1 = numpy.matmul(r1, curr_weights)
            if activation == "relu":
                r1 = relu(r1)
            elif activation == "sigmoid":
                r1 = sigmoid(r1)
        predicted_label = numpy.where(r1 == numpy.max(r1))[0][0]
        predictions[sample_idx] = predicted_label
    correct_predictions = numpy.where(predictions == data_outputs)[0].size
    accuracy = (correct_predictions/data_outputs.size)*100
    return accuracy, predictions
