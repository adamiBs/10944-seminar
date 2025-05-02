def display_image(image, title):
    import matplotlib.pyplot as plt
    
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def display_features(features, title, feature_names=None):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 4))
    x = range(len(features))
    plt.bar(x, features)
    
    if feature_names:
        plt.xticks(x, feature_names, rotation=45)
    else:
        plt.xlabel("Features")
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

def load_digits_dataset():
    from sklearn.datasets import load_digits
    digits = load_digits()
    return digits.data, digits.images

def load_iris_dataset():
    from sklearn.datasets import load_iris
    iris = load_iris()
    return iris.data, iris.feature_names, iris.target

def get_sample_image(images, sample_index):
    return images[sample_index]

def get_sample_features(data, sample_index):
    return data[sample_index]

def visualize_3d_scatter(data, target, title, save_path=None, features_to_use=[0, 1, 2]):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['r', 'g', 'b']
    markers = ['o', '^', 's']
    classes = ['Setosa', 'Versicolor', 'Virginica']
    
    for i in range(3):
        idx = (target == i)
        x = data[idx, features_to_use[0]]
        y = data[idx, features_to_use[1]]
        z = data[idx, features_to_use[2]]
        
        ax.scatter(x, y, z, c=colors[i], marker=markers[i], label=classes[i], s=30, alpha=0.7)
    
    from sklearn.datasets import load_iris
    feature_names = load_iris().feature_names
    ax.set_xlabel(feature_names[features_to_use[0]])
    ax.set_ylabel(feature_names[features_to_use[1]])
    ax.set_zlabel(feature_names[features_to_use[2]])
    
    ax.set_title(title)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.tight_layout()
    plt.show()