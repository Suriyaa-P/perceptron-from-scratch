from src.perceptron import Perceptron

and_data = [
    ([0,0],0),
    ([1,1],1),
    ([0,1],0),
    ([1,0],0)
]

or_data = [
    ([0,0],0),
    ([1,1],1),
    ([1,0],1),
    ([0,1],1)
]

not_data = [
    ([0],1),
    ([1],0)
    ]

print("---AND GATE---")
and_gate = Perceptron(2)
and_gate.train(training_data=and_data,epochs=100)

for inputs,_ in and_data:
    print(f"{inputs} : {and_gate.predict(inputs)}")

print("---OR GATE---")
or_gate = Perceptron(2)
or_gate.train(or_data,100)

for input,_ in or_data:
    print(f"{inputs}:{or_gate.predict(input)}")

print("---NOT GATE---")
not_gate = Perceptron(1)
not_gate.train(not_data)
for input,_ in not_data:
    print(f"{inputs}:{or_gate.predict(input)}")