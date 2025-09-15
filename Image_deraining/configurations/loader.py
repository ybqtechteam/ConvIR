import yaml, os

class YAMLLoader:
    
    def __init__(self, config_path="configurations/settings.yml"):
        with open(config_path, 'r') as file:
            settings = yaml.safe_load(file)
        
        for key, value in settings.items():
            setattr(self, key, value)


if __name__ == '__main__':
    settings = YAMLLoader(config_path="configurations/concat_train_on_all_benchmark.yml")
    
    for key, value in settings.__dict__.items():
        print(f"{key}: {value}")