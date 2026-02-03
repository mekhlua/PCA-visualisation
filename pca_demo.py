# PCA VISUALIZATION TOOL
# Created by Mekhluqat Abdulwehab - ASTU CS Student

print("=" * 50)
print("🔮 PCA VISUALIZATION DEMO")
print("👨‍💻 By: Mekhluqat Abdulwehab")
print("🎓 ASTU Computer Science Student")
print("=" * 50)

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

print("\n📊 Loading Iris Dataset...")
# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

print(f"✅ Dataset loaded!")
print(f"   Samples: {X.shape[0]}")
print(f"   Features: {X.shape[1]}")
print(f"   Feature names: {iris.feature_names}")
print(f"   Target names: {iris.target_names}")

print("\n🎯 Applying PCA...")
# Apply PCA (reduce to 2 dimensions)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("✅ PCA completed!")
print(f"   Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"   Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

print("\n🎨 Creating visualization...")
# Create figure
plt.figure(figsize=(12, 10))

# Plot PCA results
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=y, cmap='viridis', 
                     s=100, alpha=0.7, 
                     edgecolors='black', linewidth=1)

# Add labels and title
plt.xlabel('Principal Component 1', fontsize=14)
plt.ylabel('Principal Component 2', fontsize=14)
plt.title('PCA Visualization - Iris Dataset\nBy Mekhluqat Abdulwehab (ASTU)', 
          fontsize=16, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Flower Type', fontsize=12)

# Add grid
plt.grid(True, alpha=0.3, linestyle='--')

# Add text box with info
info_text = f"""Dataset: Iris
Samples: {X.shape[0]}
Features: {X.shape[1]}
Variance explained: {sum(pca.explained_variance_ratio_):.1%}
PC1: {pca.explained_variance_ratio_[0]:.1%}
PC2: {pca.explained_variance_ratio_[1]:.1%}"""

plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

print("\n💾 Saving plot...")
# Save the plot
plt.savefig('pca_result.png', dpi=300, bbox_inches='tight', facecolor='white')

print("✅ Plot saved as 'pca_result.png'")
print("\n📈 Displaying plot...")
plt.tight_layout()
plt.show()

print("=" * 50)
print("🎉 PCA VISUALIZATION COMPLETED SUCCESSFULLY!")
print("⭐ Check your folder for 'pca_result.png'")
print("=" * 50)
