import random

class Perceptron:
    def __init__(self,n_inps):
        self.weights = [random.uniform(-1,1) for _ in range(n_inps)]
        self.bias = random.uniform(-1,1)
        self.learning_rate = 0.1

    def predict(self,inputs):
        weighted_sum = sum((wi*xi for wi,xi in zip(self.weights,inputs)),self.bias)
        return 1 if weighted_sum >= 0 else 0

    def error(self,y,y_hat):
        cost = (y - y_hat)
        return cost

    def learning_rule(self,input,error):
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * input[i]
        self.bias += self.learning_rate * error

    def train(self,training_data,epochs=100):
        for epoch in range(epochs):
            total_errors = 0
            for x,y in training_data:
                y_hat = self.predict(x)
                error = self.error(y=y,y_hat=y_hat)

                if error != 0:
                    self.learning_rule(input=x,error=error)
                    total_errors += 1
            if total_errors == 0:
                print(f"Converged at {epoch+1} epochs") 
                return f"Converged at epoch {epoch + 1}";
        return f"Do not converged after {epochs} epochs"

