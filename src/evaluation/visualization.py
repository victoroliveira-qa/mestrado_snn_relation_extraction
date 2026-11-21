import numpy as np


def save_reductions(embeddings, labels, class_names, config, seed, output):
    if not config.get("enabled",True) or len(embeddings)<4:return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng=np.random.default_rng(seed)
    sel=rng.choice(len(embeddings),size=min(len(embeddings),int(config.get("sample_size",1000))),replace=False)
    data,classes=embeddings[sel],labels[sel]
    def draw(points,method):
        fig,ax=plt.subplots(figsize=(10,7))
        for cid,name in enumerate(class_names):
            mask=classes==cid
            if mask.any():ax.scatter(points[mask,0],points[mask,1],label=name,s=14,alpha=.7)
        ax.set_title(method);ax.legend(loc="upper left",bbox_to_anchor=(1.02,1),fontsize=8)
        fig.tight_layout();fig.savefig(output/f"{method.lower()}.png",dpi=160);plt.close(fig)
    if config.get("tsne",True) and len(data)>3:
        from sklearn.manifold import TSNE
        perplexity=min(30,max(2,(len(data)-1)//3))
        if perplexity<len(data):draw(TSNE(n_components=2,random_state=seed,perplexity=perplexity,init="pca").fit_transform(data),"TSNE")
    if config.get("pacmap",True):
        try:
            import pacmap
            draw(pacmap.PaCMAP(n_components=2,random_state=seed).fit_transform(data),"PaCMAP")
        except ImportError:
            print("PaCMAP skipped: pacmap not installed")
