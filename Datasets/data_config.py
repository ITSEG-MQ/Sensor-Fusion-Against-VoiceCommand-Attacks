BASE_DIR = './'

traffic_sign_config = {
    'inputCNNshape': (64, 64, 1),
    'inputMLPshape': (1000,),
    'data_dir': 'Train',
    #'classes': ['go', 'right','stop', 'left', "rightattackgo", "rightattackstop", "rightattackleft", "goattackright", "goattackleft", "goattackstop", "leftattackstop", "leftattackright","leftattackgo","stopattackgo","stopattackright","stopattackleft"],
    #'classes': ['go', 'right','stop', 'left'],
    #'classes': ['go', 'right', 'stop', 'left', 'goanomaly', 'stopanomaly','rightanomaly', 'leftanomaly'],
    'classes': ['go', 'right','stop', 'left','anomaly'],
    'nb_classes': 5,
}

data_constants = {
    'traffic_sign': traffic_sign_config
}


