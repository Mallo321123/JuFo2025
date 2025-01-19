import pickle

def save_model(network):
    with open('saved_model.pickle','wb') as file:
        pickle.dump(network, file)