import json

# 保存预处理参数
preprocess_config = {
    'input_size': 224,
    'normalization': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    },
    'train_augmentation': [
        'RandomResizedCrop: size=224, scale=(0.8,1.0)',
        'RandomRotation: degrees=10',
        'RandomHorizontalFlip: p=0.3',
        'ColorJitter: brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1',
        'RandomAffine: degrees=5, translate=0.1, scale=(0.9,1.1)',
        'GaussianBlur: kernel=3, sigma=(0.1,1.0)'
    ],
    'val_test_augmentation': [
        'Resize: 224x224'
    ]
}

with open('../../preprocessing_config.json', 'w', encoding='utf-8') as f:
    json.dump(preprocess_config, f, indent=2, ensure_ascii=False)

print("✅ 预处理配置已保存到 preprocessing_config.json")
