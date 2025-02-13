class Trainer:
    def __init__(self, model, CFG, is_ensemble=False, save_preds=False):
        self.model = model
        self.config = CFG
        self.is_ensemble = is_ensemble
        self.save_preds = save_preds

    def fit_predict(self, X, y, X_test):
        print(f'Training {self.model.__class__.__name__}\n')
        
        scores = []        
        coeffs = np.zeros((1, X.shape[1]))
        oof_preds = np.zeros(len(X), dtype=np.float32)
        test_preds = np.zeros(len(X_test), dtype=np.float32)
        
        skf = KFold(n_splits=self.config.n_folds, random_state=self.config.seed, shuffle=True)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            _X_test = X_test.copy()
            if not self.is_ensemble:
                te_cols = ['brand', 'transmission', 'ext_col', 'int_col']
                te = TargetEncoder()
                X_train[te_cols] = te.fit_transform(X_train[te_cols], y_train)
                X_val[te_cols] = te.transform(X_val[te_cols])
                _X_test[te_cols] = te.transform(_X_test[te_cols])
            
            model = clone(self.model)
            model.fit(X_train, y_train)
            
            if self.is_ensemble:
                coeffs += model.coef_ / self.config.n_folds
            
            y_preds = model.predict(X_val)
            oof_preds[val_idx] = y_preds
            
            temp_test_preds = model.predict(_X_test)
            test_preds += temp_test_preds / self.config.n_folds
            
            score = mean_squared_error(y_val, y_preds, squared=False)
            scores.append(score)
            
            del model, X_train, y_train, X_val, y_val, y_preds, temp_test_preds, _X_test
            gc.collect()
            
            print(f'--- Fold {fold_idx + 1} - RMSE: {round(score)}')
            
        overall_score = mean_squared_error(y, oof_preds, squared=False)
        
        print(f'\nOverall RMSE: {round(overall_score)} ± {round(np.std(scores))}')
        
        if self.save_preds:
            self._save_preds(oof_preds, overall_score, 'oof')
            self._save_preds(test_preds, overall_score, 'test')
        
        if self.is_ensemble:
            return oof_preds, test_preds, scores, coeffs[0]
        else:
            return oof_preds, test_preds, scores
        
    def _save_preds(self, preds, cv_score, name):
        model_name = self.model.__class__.__name__.lower().replace('regressor', '')
        with open(f'{name}_preds/{model_name}_{name}_preds_{round(cv_score)}.pkl', 'wb') as f:
            pickle.dump(preds, f)