from pretrain_model import pretrain_kg_model


def pretrain(config):
    if config['type'] == 'kg':
        pretrain_kg_model.pretrain_kg(config)
    else:
        pass




