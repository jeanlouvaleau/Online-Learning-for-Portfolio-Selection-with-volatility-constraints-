import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from tqdm import tqdm



#  projection on simplex of a vector v
def project_to_simplex(v):
    
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

# quadratic program solver for markowitz portfolio
def solve_qp(mu_est, Sigma_est, risk_aversion):
    """
    Résout: Max w^T*mu - (lambda/2) * w^T*Sigma*w
    Sujet à: sum(w)=1, w>=0
    """
    N = len(mu_est)
    w = cp.Variable(N)
    gamma = cp.Parameter(nonneg=True, value=risk_aversion)
    # we add a small regularization to Sigma to ensure it's positive definite
    Sigma_est = Sigma_est + np.eye(N) * 1e-6

    # Objectif : Maximiser le rendement ajusté du risque
    # Note: cvxpy minimise, donc on minimise -(Rendement - Penalité)
    ret = mu_est @ w
    risk = cp.quad_form(w, cp.psd_wrap(Sigma_est))
    prob = cp.Problem(cp.Maximize(ret - (gamma / 2) * risk), 
                      [cp.sum(w) == 1, w >= 0])
    
    try:
        prob.solve(solver=cp.OSQP, warm_start=True) 
        return w.value
    except Exception as e:
        print("failed to compute markowitz solution",e)




def get_variance(w, V):
        return np.dot(w, np.dot(V, w))


def calculate_annualized_sharpe_ratio(wealth_series, risk_free_rate=0.0, periods_per_year=252):

    returns = pd.Series(wealth_series).pct_change().dropna()
    mean_return = returns.mean()
    std_return = returns.std()
    # Formule : (Moyenne - Rf) / Volatilité * Racine(Périodes)
    # Note: On suppose Rf journalier ~ 0 pour simplifier
    sharpe = (mean_return - risk_free_rate) / std_return
    annualized_sharpe = sharpe * np.sqrt(periods_per_year)
    
    return annualized_sharpe


def get_rolling_sharpe(wealth_array, window, rf=0.0):

    s = pd.Series(wealth_array)

    returns = s.pct_change()

    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    
    rolling_sharpe = (roll_mean - rf) / roll_std * np.sqrt(252)
    
    return rolling_sharpe




def plot_sharpe_vs_learning_rate(learning_rates, returns, covs, window=252, lambda_reg=25):

    T, N = returns.shape
    rebalance_freq = 1 
    decay = 0.9      
    epsilon = 1e-8

    # Dictionnaire résultats
    results = {
        'OGD (Adaptive LR based on past gradients)': [],
        'EG (Adaptive LR based on past gradients)': [],
        'Optimistic OGD (Adaptive LR based on past gradients)': [],
        'Optimistic EG (Adaptive LR based on past gradients)': [],
        'Optimistic OGD (LR based on past forecasts and gradients)': [], 
        'Optimistic EG (LR based on past forecasts and gradients)': []   
    }

    # --- BOUCLE SUR LES LEARNING RATES ---
    for lr in tqdm(learning_rates):
        
        # Initialisation
        w_ogd_std = np.ones(N)/N; wealth_ogd_std = [1.0]; cum_grad_ogd_std = np.zeros(N); ma_sq_ogd_std = 0.0
        w_eg_std  = np.ones(N)/N; wealth_eg_std  = [1.0]; cum_grad_eg_std  = np.zeros(N); ma_sq_eg_std  = 0.0
        
        w_ogd_opt_g = np.ones(N)/N; wealth_ogd_opt_g = [1.0]; cum_grad_ogd_opt_g = np.zeros(N); ma_sq_ogd_opt_g = 0.0
        w_eg_opt_g  = np.ones(N)/N; wealth_eg_opt_g  = [1.0]; cum_grad_eg_opt_g  = np.zeros(N); ma_sq_eg_opt_g  = 0.0
        
        w_ogd_opt_p = np.ones(N)/N; wealth_ogd_opt_p = [1.0]; cum_grad_ogd_opt_p = np.zeros(N); sum_sq_err_ogd = 0.0
        w_eg_opt_p  = np.ones(N)/N; wealth_eg_opt_p  = [1.0]; cum_grad_eg_opt_p  = np.zeros(N); sum_sq_err_eg  = 0.0

        # --- BOUCLE TEMPORELLE ---
        for t in range(window, T):

            scale_boost = (t - window + 1) if (t > window) else 1.0

            r_t = returns[t]
            V_t = covs[t]

            # A. Performance
            wealth_ogd_std.append(wealth_ogd_std[-1] * (1 + np.dot(w_ogd_std, r_t)))
            wealth_eg_std.append(wealth_eg_std[-1]   * (1 + np.dot(w_eg_std, r_t)))
            wealth_ogd_opt_g.append(wealth_ogd_opt_g[-1] * (1 + np.dot(w_ogd_opt_g, r_t)))
            wealth_eg_opt_g.append(wealth_eg_opt_g[-1]   * (1 + np.dot(w_eg_opt_g, r_t)))
            wealth_ogd_opt_p.append(wealth_ogd_opt_p[-1] * (1 + np.dot(w_ogd_opt_p, r_t)))
            wealth_eg_opt_p.append(wealth_eg_opt_p[-1]   * (1 + np.dot(w_eg_opt_p, r_t)))

            # B. Updates
            if t % rebalance_freq == 0:
                mu_pred = np.mean(returns[t-window:t], axis=0)
                sigma_pred = np.mean(covs[t-window:t], axis=0)

                # 1. STANDARD
                g = r_t - lambda_reg * (V_t @ w_ogd_std)
                cum_grad_ogd_std += g
                ma_sq_ogd_std = decay * ma_sq_ogd_std + (1-decay) * np.sum(g**2)
                eta = lr / (np.sqrt(ma_sq_ogd_std) + epsilon)
                w_ogd_std = project_to_simplex(eta * cum_grad_ogd_std)

                g = r_t - lambda_reg * (V_t @ w_eg_std)
                cum_grad_eg_std += g
                ma_sq_eg_std = decay * ma_sq_eg_std + (1-decay) * np.sum(g**2)
                eta = lr / (np.sqrt(ma_sq_eg_std) + epsilon)
                score = eta * cum_grad_eg_std
                w_eg_std = np.exp(score - np.max(score)) / np.sum(np.exp(score - np.max(score)))

                # 2. OPTIMISTIC (GRAD LR)
                g = r_t - lambda_reg * (V_t @ w_ogd_opt_g)
                cum_grad_ogd_opt_g += g
                ma_sq_ogd_opt_g = decay * ma_sq_ogd_opt_g + (1-decay) * np.sum(g**2)
                eta = lr / (np.sqrt(ma_sq_ogd_opt_g) + epsilon)
                g_pred = mu_pred - lambda_reg * (sigma_pred @ w_ogd_opt_g)
                w_ogd_opt_g = project_to_simplex(eta * (cum_grad_ogd_opt_g + scale_boost*g_pred))

                g = r_t - lambda_reg * (V_t @ w_eg_opt_g)
                cum_grad_eg_opt_g += g
                ma_sq_eg_opt_g = decay * ma_sq_eg_opt_g + (1-decay) * np.sum(g**2)
                eta = lr / (np.sqrt(ma_sq_eg_opt_g) + epsilon)
                g_pred = mu_pred - lambda_reg * (sigma_pred @ w_eg_opt_g)
                score = eta * (cum_grad_eg_opt_g + scale_boost*g_pred)
                w_eg_opt_g = np.exp(score - np.max(score)) / np.sum(np.exp(score - np.max(score)))

                # 3. OPTIMISTIC (PRED LR)
                g = r_t - lambda_reg * (V_t @ w_ogd_opt_p)
                g_pred_curr = mu_pred - lambda_reg * (sigma_pred @ w_ogd_opt_p)
                err_sq = np.sum((g - g_pred_curr)**2)
                sum_sq_err_ogd += err_sq
                eta = lr / (np.sqrt(sum_sq_err_ogd) + epsilon)
                cum_grad_ogd_opt_p += g
                w_ogd_opt_p = project_to_simplex(eta * (cum_grad_ogd_opt_p + scale_boost*g_pred_curr))

                g = r_t - lambda_reg * (V_t @ w_eg_opt_p)
                g_pred_curr = mu_pred - lambda_reg * (sigma_pred @ w_eg_opt_p)
                err_sq = np.sum((g - g_pred_curr)**2)
                sum_sq_err_eg += err_sq
                eta = lr / (np.sqrt(sum_sq_err_eg) + epsilon)
                cum_grad_eg_opt_p += g
                score = eta * (cum_grad_eg_opt_p + scale_boost*g_pred_curr)
                w_eg_opt_p = np.exp(score - np.max(score)) / np.sum(np.exp(score - np.max(score)))

        # Fin boucle temporelle
        results['OGD (Adaptive LR based on past gradients)'].append(calculate_annualized_sharpe_ratio(wealth_ogd_std))
        results['EG (Adaptive LR based on past gradients)'].append(calculate_annualized_sharpe_ratio(wealth_eg_std))
        results['Optimistic OGD (Adaptive LR based on past gradients)'].append(calculate_annualized_sharpe_ratio(wealth_ogd_opt_g))
        results['Optimistic EG (Adaptive LR based on past gradients)'].append(calculate_annualized_sharpe_ratio(wealth_eg_opt_g))
        results['Optimistic OGD (LR based on past forecasts and gradients)'].append(calculate_annualized_sharpe_ratio(wealth_ogd_opt_p))
        results['Optimistic EG (LR based on past forecasts and gradients)'].append(calculate_annualized_sharpe_ratio(wealth_eg_opt_p))

    # --- PLOTTING ---
    plt.figure(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results))) 

    # --- [NOUVEAU] BLOC D'ANALYSE DES LEANING RATES ---
    print("\n" + "="*80)
    print(f"{'STRATÉGIE':<60} | {'LR CIBLE':<10} | {'SHARPE':<10}")
    print("="*80)
    
    target_sharpe = 0.95
    
    for i, (name, sharpes) in enumerate(results.items()):
        sharpes_arr = np.array(sharpes)
        lrs_arr = np.array(learning_rates)
        
        if len(sharpes_arr) == 0: continue
        
        max_sharpe = np.max(sharpes_arr)
        idx_max = np.argmax(sharpes_arr)
        
        # Logique de sélection
        if max_sharpe < target_sharpe:
            # Cas 1 : On n'atteint pas 0.95 -> On prend le Max
            selected_lr = lrs_arr[idx_max]
            selected_sharpe = max_sharpe
            note = "(Max)"
        else:
            # Cas 2 : On dépasse 0.95 -> On cherche le LR le plus proche de 0.95
            # np.argmin renvoie le premier index (donc le plus petit LR si trié) qui minimise la différence
            idx_closest = np.argmin(np.abs(sharpes_arr - target_sharpe))
            selected_lr = lrs_arr[idx_closest]
            selected_sharpe = sharpes_arr[idx_closest]
            note = "(~0.95)"

        print(f"{name:<60} | {selected_lr:.4f}     | {selected_sharpe:.4f} {note}")

        # --- Plotting standard ---
        plt.plot(learning_rates, sharpes, label=name, linewidth=2, alpha=0.8, color=colors[i])
        
        # Marquer le max sur le graph
        plt.scatter(lrs_arr[idx_max], max_sharpe, color=colors[i], s=100, zorder=10)
        plt.annotate(f'{max_sharpe:.2f}', 
                     xy=(lrs_arr[idx_max], max_sharpe), 
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=9, color=colors[i], fontweight='bold')

    print("="*80 + "\n")

    plt.xlabel('Numerator of the learning rate', fontsize=12)
    plt.ylabel('Annualized Sharpe Ratio', fontsize=12)
    plt.title('Sensitivity Analysis: Sharpe Ratio vs Learning Rate', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_strategy_performance(strategies, dates, window, weight_histories=None, assets=None):
    
    # Alignement des dates pour l'affichage (on suppose que wealth commence après 'window')
    plot_dates = dates[window:]
    
    # ==========================================
    # 1. CUMULATIVE WEALTH PLOT
    # ==========================================
    plt.figure(figsize=(12, 6))
    
    for name, wealth in strategies.items():
        # Sécurité : alignement des dimensions
        # Le wealth array a souvent 1 élément de plus (le 1.0 initial) ou correspond exactement
        if len(wealth) > len(plot_dates):
            # On suppose wealth = [1.0, w1, w2...] et plot_dates correspond à t=1..T
            y_values = wealth[1:] 
        else:
            y_values = wealth
            
        # Truncate to match shortest length just in case
        limit = min(len(plot_dates), len(y_values))
        plt.plot(plot_dates[:limit], y_values[:limit], label=name, linewidth=1.5)

    plt.title("Wealth Evolution Comparison")
    plt.ylabel("Cumulative Wealth")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # ==========================================
    # 2. SHARPE RATIO BAR CHART
    # ==========================================
    sharpe_results = {}
    for name, wealth in strategies.items():
        sharpe_results[name] = calculate_annualized_sharpe_ratio(wealth)
        
    plt.figure(figsize=(12, 6))
    names = list(sharpe_results.keys())
    values = list(sharpe_results.values())
    
    bars = plt.bar(names, values, alpha=0.7, edgecolor='black', color=plt.cm.tab10(np.arange(len(names))))

    plt.title("Annualized Sharpe Ratio by Strategy")
    plt.ylabel("Sharpe Ratio")
    plt.axhline(0, color='black', linewidth=0.8) 
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    # Valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        offset = 0.05 if height >= 0 else -0.15
        plt.text(bar.get_x() + bar.get_width()/2., height + offset,
                 f'{height:.2f}',
                 ha='center', va='bottom' if height >=0 else 'top', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # ==========================================
    # 3. ROLLING SHARPE RATIO
    # ==========================================
    plt.figure(figsize=(12, 6))
    rolling_window_sharpe = 252 # 1 an de trading
    
    for name, wealth in strategies.items():
        sharpe_series = get_rolling_sharpe(wealth, rolling_window_sharpe)
        
        # Alignement dates / séries
        # La série rolling aura des NaN au début
        if len(plot_dates) == len(sharpe_series):
            plt.plot(plot_dates, sharpe_series, label=name, linewidth=1.5)
        else:
            # Gestion basique des décalages de taille
            limit = min(len(plot_dates), len(sharpe_series))
            plt.plot(plot_dates[-limit:], sharpe_series.iloc[-limit:], label=name, linewidth=1.5)

    plt.title(f"Rolling Annualized Sharpe Ratio ({rolling_window_sharpe}-day window)")
    plt.ylabel("Sharpe Ratio")
    plt.xlabel("Date")
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.8)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ==========================================
    # 4. WEIGHTS STACKPLOTS (Optionnel)
    # ==========================================
    if weight_histories is not None and assets is not None:
        for name, history in weight_histories.items():
            plt.figure(figsize=(10, 5))
            
            # Conversion en array numpy si liste
            hist_arr = np.array(history)
            
            # Vérification dimensions pour le plot
            # stackplot demande (x, y1, y2...) où y sont de dimension (N, T)
            # hist_arr est souvent (T, N), donc on transpose -> .T
            
            # Alignement temporel : history a souvent été rempli pendant la boucle
            # Il faut s'assurer que sa longueur matche plot_dates
            limit = min(len(plot_dates), hist_arr.shape[0])
            
            plt.stackplot(plot_dates[:limit], hist_arr[:limit].T, labels=assets)
            
            plt.title(f"Evolution of weights: {name}")
            plt.ylabel("Allocation")
            plt.xlabel("Date")
            
            # Légende à l'extérieur pour ne pas cacher le graph
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', ncol=1)
            plt.tight_layout()
            plt.show()