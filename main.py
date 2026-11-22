"""SNN Research 3.1: configurable architectures + immutable experiment runs."""
import argparse
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed
from src.utils.output import output_dirs
from src.utils.runs import create_run, write_txt_report
from src.data.loader import load_table
from src.embeddings.providers import load_vectors
from src.data.splitting import split_ids, split_official_ids
from src.data.pairing import PairSequence
from src.models.siamese import build_model, L2Normalize, AbsoluteDifference
from src.models.architectures import architecture_name, shared_encoder, build_mlp, build_snn_cosine
from src.training.episodic import train_prototypical
from src.evaluation.prototypes import compute_prototypes, classify
from src.evaluation.reports import save_metrics, save_training_curves, save_confusion
from src.evaluation.audit import build_run_audit
from src.evaluation.visualization import save_reductions
from src.inference.predictor import predict_embedding

ROOT = Path(__file__).resolve().parent


def pair_metrics(model, generator, steps, dirs, prefix):
    left, right, yt = [], [], []
    for batch in range(min(int(steps), len(generator))):
        (a,b), labels = generator[batch]
        left.append(a);right.append(b);yt.append(labels)
    a,b,yt = np.vstack(left),np.vstack(right),np.concatenate(yt).astype(np.int32)
    probabilities = model.predict([a,b], batch_size=256, verbose=0).reshape(-1)
    yp = (probabilities >= .5).astype(np.int32)
    payload = {'pair_accuracy': float(accuracy_score(yt,yp)),
               'pair_f1': float(f1_score(yt,yp,zero_division=0)),
               'pair_auc': float(roc_auc_score(yt,probabilities)) if len(np.unique(yt)) > 1 else None,
               'pair_report': classification_report(yt, yp, labels=[0,1], target_names=['different','same'], output_dict=True,zero_division=0),
               'confusion_matrix': confusion_matrix(yt,yp,labels=[0,1]).tolist()}
    (dirs['metrics'] / f'{prefix}_pair_metrics.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    pd.DataFrame({'true_pair':yt, 'predicted_pair':yp, 'prob_same':probabilities}).to_csv(dirs['predictions']/f'{prefix}_pairs.csv', index=False)
    return payload


def multiclass_metrics(y, pred, labels, dirs, split):
    return save_metrics(y, pred, labels, dirs['metrics'], split)


def save_results(df, row_ids, y, pred, scores, class_names, dirs, id_column=None):
    data = {'original_row_id':row_ids, 'true_label':class_names[y], 'predicted_label':class_names[pred]}
    if id_column and id_column in df.columns:
        data[id_column] = df.loc[row_ids, id_column].to_numpy()
    results = pd.DataFrame(data)
    for idx, name in enumerate(class_names):
        results[f'score_{name}'] = scores[:,idx]
    results.to_csv(dirs['predictions']/'predictions.csv',index=False)


def check_class_coverage(y_train, y_val, y_test):
    missing = sorted((set(y_val) | set(y_test)) - set(y_train))
    if missing:
        raise ValueError(f'Class labels only in validation/test, not train: {missing}. '
                         'Choose a split with training coverage; do not evaluate missing-class prototypes.')


def train(config_path, component_overrides=None):
    cfg = load_config(config_path, component_overrides=component_overrides)  # Expands components and checks required sections first.
    name = architecture_name(cfg)
    seed = int(cfg['experiment']['seed']); set_seed(seed)
    out_base = resolve_path(cfg['experiment'].get('output_dir','results'), ROOT)
    run, dirs = create_run(out_base, cfg['experiment']['name'])
    (dirs['configs']/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False,allow_unicode=True),encoding='utf-8')
    (dirs['configs']/'source_config_path.txt').write_text(str(Path(config_path).resolve()),encoding='utf-8')
    metrics = {}
    try:
        dcfg = cfg['dataset']
        df = load_table(resolve_path(dcfg['path'], ROOT), dcfg['label_column'], dcfg.get('adapter'), dcfg.get('column_map'))
        if dcfg.get('expected_rows') and len(df) != int(dcfg['expected_rows']):
            raise ValueError('Dataset row count differs from expected_rows')
        embedding_audit = {}
        all_vec = load_vectors(cfg['embedding'],df,ROOT,audit=embedding_audit)
        valid = df[dcfg['label_column']].notna().to_numpy()
        X = np.asarray(all_vec[valid], dtype=np.float32)
        original_ids = df.loc[valid, '_original_row_id'].to_numpy()
        labels = LabelEncoder()
        y = labels.fit_transform(df.loc[valid,dcfg['label_column']].astype(str).to_numpy())
        if len(X) < 10 or len(labels.classes_) < 2:
            raise ValueError('Training requires at least 10 labeled rows and two classes')
        if cfg['split'].get('strategy') == 'official':
            if '_source_split' not in df.columns:
                raise ValueError('Official split requires an _source_split column in the dataset')
            split = split_official_ids(y, df.loc[valid, '_source_split'].to_numpy(), cfg['split'], seed)
        else:
            split = split_ids(y,cfg['split'],seed)
        tr,va,te = (split[k] for k in ('train','validation','test'))
        check_class_coverage(y[tr],y[va],y[te])
        np.savez_compressed(dirs['splits']/'split_indices.npz', **{k:original_ids[v] for k,v in split.items()})
        np.save(dirs['models']/'label_classes.npy',labels.classes_)
        tc = cfg['training']; mc = cfg['model']; lr = float(tc['learning_rate'])
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(tc['early_stopping_patience']), restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=int(tc['reduce_lr_patience']), factor=.5)]
        if name in ('mlp','mlp_prototype'):
            model, encoder = build_mlp(X.shape[1],len(labels.classes_),mc,lr)
            history = model.fit(X[tr],y[tr], validation_data=(X[va],y[va]),
                                epochs=int(tc['epochs']),batch_size=int(tc['batch_size']),callbacks=callbacks,verbose=2)
            model.save(dirs['models']/'model.keras');encoder.save(dirs['models']/'encoder.keras')
            if name == 'mlp':
                va_prob = model.predict(X[va],verbose=0); te_prob=model.predict(X[te],verbose=0)
                va_pred=va_prob.argmax(1);pred=te_prob.argmax(1); scores=te_prob
                visual=encoder.predict(X[te],verbose=0)
            else:
                va_pred,pred,scores,visual = prototype_predictions(encoder,X,y,tr,va,te,dirs,len(labels.classes_))
        elif name == 'prototype':
            encoder = shared_encoder(X.shape[1],mc,prefix='prototype')
            history = train_prototypical(encoder,X[tr],y[tr],X[va],y[va],cfg,dirs)
            encoder.save(dirs['models']/'encoder.keras')
            va_pred,pred,scores,visual = prototype_predictions(encoder,X,y,tr,va,te,dirs,len(labels.classes_))
        else:
            pairtrain = PairSequence(X[tr],y[tr],tc['batch_size'],tc['steps_per_epoch'],seed,tc.get('positive_fraction',.5))
            pairval = PairSequence(X[va],y[va],tc['batch_size'],tc['validation_steps'],seed+1000,tc.get('positive_fraction',.5))
            if name == 'snn':
                model,encoder=build_snn_cosine(X.shape[1],mc,lr)
            else:
                model,encoder=build_model(X.shape[1],mc,lr)  # Unmodified Step-3 SNN+MLP model.
            history=model.fit(pairtrain,validation_data=pairval,epochs=int(tc['epochs']),callbacks=callbacks,verbose=2)
            model.save(dirs['models']/'siamese.keras')
            encoder.save(dirs['models']/'encoder.keras')
            metrics['validation']=pair_metrics(model,pairval,tc['validation_steps'],dirs,'validation')
            pairtest=PairSequence(X[te],y[te],tc['batch_size'],tc.get('test_steps',tc['validation_steps']),seed+2000,tc.get('positive_fraction',.5))
            metrics['test']=pair_metrics(model,pairtest,tc.get('test_steps',tc['validation_steps']),dirs,'test')
            if name == 'snn_prototype':
                # Keep legacy Step-3 multiclass inference separate from pair evaluation.
                va_pred,pred,scores,visual=prototype_predictions(encoder,X,y,tr,va,te,dirs,len(labels.classes_))
                metrics['validation_multiclass']=multiclass_metrics(y[va],va_pred,labels.classes_,dirs,'validation_multiclass')
                metrics['test_multiclass']=multiclass_metrics(y[te],pred,labels.classes_,dirs,'test_multiclass')
                save_results(df,original_ids[te],y[te],pred,scores,labels.classes_,dirs,dcfg.get('id_column'))
                save_confusion(y[te],pred,labels.classes_,dirs['figures'])
                save_reductions(visual,y[te],labels.classes_,cfg.get('visualization',{}),seed,dirs['figures'])
        if name in ('mlp','mlp_prototype','prototype'):
            metrics['validation']=multiclass_metrics(y[va],va_pred,labels.classes_,dirs,'validation')
            metrics['test']=multiclass_metrics(y[te],pred,labels.classes_,dirs,'test')
            save_results(df,original_ids[te],y[te],pred,scores,labels.classes_,dirs,dcfg.get('id_column'))
            save_confusion(y[te],pred,labels.classes_,dirs['figures'])
            save_reductions(visual,y[te],labels.classes_,cfg.get('visualization',{}),seed,dirs['figures'])
        save_training_curves(history,dirs['figures'],dirs['logs'])
        metadata = {'created_utc': datetime.now(timezone.utc).isoformat(), 'architecture':name,
                    'dataset':str(dcfg['path']), 'embedding':cfg['embedding'],
                    'labeled_rows':int(len(X)), 'embedding_dim':int(X.shape[1]),
                    'split_sizes':{k:len(v) for k,v in split.items()},
                    'split_strategy':cfg['split'].get('strategy','random'),
                    'dataset_adapter':dcfg.get('adapter','tabular'),
                    'dataset_sha256':hashlib.sha256(resolve_path(dcfg['path'], ROOT).read_bytes()).hexdigest(),
                    'notes':'Pair binary scores and multiclass prototype results are distinct tasks. Prototype scores are cosine similarities, not probabilities.'}
        (dirs['logs']/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
        audit = build_run_audit(history, y, split, labels.classes_, embedding_audit)
        (dirs['logs']/'experimental_audit.json').write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
        report=write_txt_report(dirs,cfg,metrics,run,audit=audit)
        print(f'RUN={run}\nRELATORIO={report}')
        return run
    except Exception as exc:
        (dirs['logs']/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write_txt_report(dirs,cfg,metrics,run,status='FALHOU',extra=[f'Erro: {type(exc).__name__}: {exc}'])
        raise


def prototype_predictions(encoder,X,y,tr,va,te,dirs,class_count):
    train_proj = encoder.predict(X[tr],batch_size=256,verbose=0)
    val_proj=encoder.predict(X[va],batch_size=256,verbose=0)
    test_proj=encoder.predict(X[te],batch_size=256,verbose=0)
    prototypes,missing=compute_prototypes(train_proj,y[tr],class_count)
    if missing:
        raise ValueError(f'Cannot evaluate class prototypes: missing train classes {missing}')
    np.save(dirs['models']/'prototypes.npy',prototypes)
    (dirs['models']/'missing_train_classes.json').write_text('[]',encoding='utf-8')
    va_pred,_=classify(val_proj,prototypes,missing)
    pred,scores=classify(test_proj,prototypes,missing)
    return va_pred,pred,scores,test_proj


def infer(config_path, experiment_dir, row_id, other_row_id=None):
    exp=Path(experiment_dir).resolve()
    # Always load saved run configuration to avoid mismatching model/data after changes.
    saved=exp/'configs'/'config.yaml'
    if not saved.is_file():
        raise FileNotFoundError(f'Run config not found: {saved}')
    cfg=load_config(saved)
    name=architecture_name(cfg)
    dirs=output_dirs(exp,create=False)
    dcfg = cfg['dataset']
    df=load_table(resolve_path(dcfg['path'], ROOT),dcfg['label_column'],dcfg.get('adapter'),dcfg.get('column_map'))
    if not (0<=row_id<len(df)):
        raise IndexError('row-id outside dataset')
    X=load_vectors(cfg['embedding'],df,ROOT)
    if name in ('snn','snn_mlp'):
        if other_row_id is None:
            raise ValueError('Pair-only architecture requires --other-row-id')
        if not (0<=other_row_id<len(df)):
            raise IndexError('other-row-id outside dataset')
        model=tf.keras.models.load_model(dirs['models']/'siamese.keras')
        prob=float(model.predict([X[row_id:row_id+1], X[other_row_id:other_row_id+1]],verbose=0).reshape(-1)[0])
        result={'architecture':name,'row_id':row_id,'other_row_id':other_row_id,'similarity_score':prob,'same_relation':bool(prob>=.5)}
    elif name=='mlp':
        model=tf.keras.models.load_model(dirs['models']/'model.keras')
        labels=np.load(dirs['models']/'label_classes.npy',allow_pickle=False)
        probs=model.predict(X[row_id:row_id+1],verbose=0)[0]
        order=np.argsort(probs)[::-1][:5]
        result={'architecture':name,'row_id':row_id,'predicted_label':str(labels[order[0]]),'top_k':[{'label':str(labels[i]),'probability':float(probs[i])} for i in order]}
    else:
        encoder=tf.keras.models.load_model(dirs['models']/'encoder.keras')
        labels=np.load(dirs['models']/'label_classes.npy',allow_pickle=False)
        prototypes=np.load(dirs['models']/'prototypes.npy')
        missing=json.loads((dirs['models']/'missing_train_classes.json').read_text())
        result=predict_embedding(X[row_id],encoder,prototypes,labels,missing,normalize_input=False)
        result['architecture']=name;result['row_id']=row_id
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='SNN Research: modular architectures, immutable experiment runs')
    parser.add_argument('command',choices=['train','infer'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--dataset', help='Override dataset preset (train only)')
    parser.add_argument('--embedding', help='Override embedding preset (train only)')
    parser.add_argument('--architecture', help='Override architecture preset (train only)')
    parser.add_argument('--experiment-dir')
    parser.add_argument('--row-id',type=int)
    parser.add_argument('--other-row-id',type=int)
    args=parser.parse_args()
    if args.command=='train':
        chosen = {k: v for k, v in [('dataset', args.dataset), ('embedding', args.embedding), ('architecture', args.architecture)] if v is not None}
        train(args.config, component_overrides=chosen)
    else:
        if args.dataset or args.embedding or args.architecture:
            parser.error('Inference uses the saved run configuration; component overrides are train-only')
        if args.experiment_dir is None or args.row_id is None:
            parser.error('infer requires --experiment-dir and --row-id')
        infer(args.config,args.experiment_dir,args.row_id,args.other_row_id)
