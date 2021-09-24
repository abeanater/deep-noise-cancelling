

def training_harness(config):
    model = pass
    loss = pass
    for epoch in range(cfg['num_epochs']):
        for i,x,y in enumerate(training):
            y_hat = model(x)
            train_loss = loss(y_hat,y)
            train_loss.backward()