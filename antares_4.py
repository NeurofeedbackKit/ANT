from pathlib import Path
import logging
import yaml
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

def rank_features(subject_id, subjects_dir, visit):

    subject_dir = subjects_dir / subject_id
    reliability_dir = subject_dir / "reliability_metrics"
    reliability_files = list(reliability_dir.glob(f"*_v{visit}.csv"))

    all_rankings = []
    for rel_file in reliability_files:
        df_r = pd.read_csv(rel_file)
        if not "deviation" in df_r.columns.tolist():
            continue

        for idx, row in df_r.iterrows():
            feat_name = row['feature_name']
            z_score = row['deviation']
            icc = row['icc']
            snr = row['snr']
            dynamic_range = row['dynamic_range']
            stationarity = row['stationarity']
            
            # Apply thresholds
            meets_criteria = (
                z_score >= 1.5 and  
                icc >= 0.6 and      
                snr >= 2.0 and      
                dynamic_range >= 0.3
            )
            
            # Compute composite score (weighted)
            # Normalize each metric to 0-1 range for fair weighting
            norm_z = min(z_score / 3.0, 1.0)  
            norm_icc = min(max(icc, 0), 1.0)
            norm_snr = min(snr / 10.0, 1.0)  
            norm_dr = min(dynamic_range / 2.0, 1.0)
            norm_stat = min(stationarity, 1.0)
            
            composite_score = (
                0.30 * norm_z +           # Clinical relevance
                0.25 * norm_icc +         # Reliability
                0.20 * norm_snr +         # Real-time detectability
                0.15 * norm_dr +          # Trainability
                0.10 * norm_stat          # Stability
            )
            
            all_rankings.append({
                'feature_name': feat_name,
                'feature_type': row['feature_type'],
                'z_score': z_score,
                'icc': icc,
                'snr': snr,
                'dynamic_range': dynamic_range,
                'stationarity': stationarity,
                'composite_score': composite_score,
                'meets_criteria': meets_criteria
            })

    df_ranked = pd.DataFrame(all_rankings)
    df_ranked = df_ranked.sort_values('composite_score', ascending=False)
    df_ranked['rank'] = range(1, len(df_ranked) + 1)

    # Select top feature
    viable_features = df_ranked[df_ranked['meets_criteria'] == True]

    if len(viable_features) > 0:
        selected_feature = viable_features.iloc[0].to_dict()
        backup_features = viable_features.iloc[1:4].to_dict('records')
        selection_status = "success"
        selection_message = f"Selected {selected_feature['feature_name']} (composite score: {selected_feature['composite_score']:.3f})"
    else:
        # Relaxed selection: just pick highest composite score
        selected_feature = df_ranked.iloc[0].to_dict()
        backup_features = df_ranked.iloc[1:4].to_dict('records')
        selection_status = "relaxed_criteria"
        selection_message = f"No features met all criteria. Selected best available: {selected_feature['feature_name']}"

    # Save ranking table
    output_dir = subject_dir / "nf_selection"
    output_dir.mkdir(exist_ok=True)

    df_ranked.to_csv(
        output_dir / f"feature_ranking_v{visit}.csv",
        index=False
    )

    # Create report dictionary
    report_dict = {
        'subject_id': subject_id,
        'visit': visit,
        'selection_status': selection_status,
        'selection_message': selection_message,
        'selected_feature': selected_feature,
        'backup_features': backup_features,
        'n_features_evaluated': len(df_ranked),
        'n_features_meeting_criteria': len(viable_features),
        'ranking_table': df_ranked
    }

    # Save JSON report
    json_report = {k: v for k, v in report_dict.items() if k != 'ranking_table'}
    json_report['selected_feature'] = {
        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
        for k, v in selected_feature.items()
    }

    with open(output_dir / f"selection_report_v{visit}.json", 'w') as f:
        json.dump(json_report, f, indent=2)

    return report_dict


def generate_selection_report_html(report_dict, subjects_dir):
    """
    Generate a beautiful HTML report for feature selection.
    """

    
    subject_id = report_dict['subject_id']
    visit = report_dict['visit']
    subject_dir = subjects_dir / subject_id
    output_dir = subject_dir / "nf_selection"
    
    selected = report_dict['selected_feature']
    backups = report_dict['backup_features']
    df_ranked = report_dict['ranking_table']
    
    # Create visualizations
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Feature Selection Report: {subject_id} - Visit {visit}", 
                fontsize=16, fontweight='bold')
    
    # 1. Top 20 features by composite score
    ax = axes[0, 0]
    top_20 = df_ranked.head(20)
    colors = ['#2ecc71' if meets else '#e74c3c' 
                for meets in top_20['meets_criteria']]
    ax.barh(range(len(top_20)), top_20['composite_score'], color=colors)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20['feature_name'], fontsize=8)
    ax.set_xlabel('Composite Score')
    ax.set_title('Top 20 Features by Composite Score')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.invert_yaxis()
    
    # Add legend

    legend_elements = [
        Patch(facecolor='#2ecc71', label='Meets all criteria'),
        Patch(facecolor='#e74c3c', label='Below threshold')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    # 2. Metric breakdown for selected feature
    ax = axes[0, 1]
    metrics = ['z_score', 'icc', 'snr', 'dynamic_range', 'stationarity']
    values = [selected[m] for m in metrics]
    thresholds = [1.5, 0.6, 2.0, 0.3, 0.7]
    
    x = np.arange(len(metrics))
    bars = ax.bar(x, values, color='#3498db', alpha=0.7)
    ax.scatter(x, thresholds, color='red', s=100, marker='_', linewidths=3, 
                label='Threshold', zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(['Z-score', 'ICC', 'SNR', 'Dyn. Range', 'Stationarity'], 
                        rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title(f'Selected Feature Metrics\n{selected["feature_name"]}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Distribution of metrics across all features
    ax = axes[1, 0]
    metric_to_plot = 'composite_score'
    ax.hist(df_ranked[metric_to_plot], bins=30, color='#95a5a6', alpha=0.7, 
            edgecolor='black')
    ax.axvline(selected[metric_to_plot], color='#2ecc71', linewidth=3, 
            label='Selected feature', linestyle='--')
    if len(backups) > 0:
        for backup in backups:
            ax.axvline(backup[metric_to_plot], color='#f39c12', linewidth=2, 
                    alpha=0.6, linestyle=':')
    ax.set_xlabel('Composite Score')
    ax.set_ylabel('Number of Features')
    ax.set_title('Distribution of Composite Scores')
    ax.legend()
    
    # 4. Scatter: ICC vs SNR (colored by composite score)
    ax = axes[1, 1]
    scatter = ax.scatter(df_ranked['icc'], df_ranked['snr'], 
                        c=df_ranked['composite_score'], 
                        cmap='viridis', s=50, alpha=0.6)
    ax.scatter(selected['icc'], selected['snr'], 
            color='#2ecc71', s=300, marker='*', 
            edgecolor='black', linewidth=2, label='Selected', zorder=5)
    
    # Mark thresholds
    ax.axhline(2.0, color='red', linestyle='--', alpha=0.3, label='SNR threshold')
    ax.axvline(0.6, color='red', linestyle='--', alpha=0.3, label='ICC threshold')
    
    ax.set_xlabel('ICC (Reliability)')
    ax.set_ylabel('SNR')
    ax.set_title('Feature Space: ICC vs SNR')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Composite Score')
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / f"selection_report_v{visit}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Neurofeedback Feature Selection - {subject_id}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 40px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .status-success {{
                background-color: #2ecc71;
                color: white;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .status-relaxed {{
                background-color: #f39c12;
                color: white;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .feature-card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .metric {{
                display: inline-block;
                margin: 10px 20px 10px 0;
            }}
            .metric-label {{
                font-weight: bold;
                color: #555;
            }}
            .metric-value {{
                font-size: 1.3em;
                color: #2c3e50;
            }}
            .good {{ color: #2ecc71; }}
            .warning {{ color: #f39c12; }}
            .bad {{ color: #e74c3c; }}
            table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                margin-top: 20px;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #667eea;
                color: white;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            img {{
                width: 100%;
                border-radius: 10px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Neurofeedback Feature Selection Report</h1>
            <h2>Subject: {subject_id} | Visit: {visit}</h2>
        </div>
        
        <div class="{'status-success' if report_dict['selection_status'] == 'success' else 'status-relaxed'}">
            <strong>Status:</strong> {report_dict['selection_message']}
        </div>
        
        <div class="feature-card">
            <h2> Selected Feature</h2>
            <h3>{selected['feature_name']}</h3>
            <p><em>Type: {selected['feature_type']}</em></p>
            
            <div class="metric">
                <span class="metric-label">Composite Score:</span>
                <span class="metric-value {'good' if selected['composite_score'] > 0.6 else 'warning' if selected['composite_score'] > 0.4 else 'bad'}">
                    {selected['composite_score']:.3f}
                </span>
            </div>
            
            <div class="metric">
                <span class="metric-label">Z-Score (deviation):</span>
                <span class="metric-value {'good' if selected['z_score'] >= 1.5 else 'bad'}">
                    {selected['z_score']:.2f}
                </span>
            </div>
            
            <div class="metric">
                <span class="metric-label">ICC (reliability):</span>
                <span class="metric-value {'good' if selected['icc'] >= 0.6 else 'bad'}">
                    {selected['icc']:.3f}
                </span>
            </div>
            
            <div class="metric">
                <span class="metric-label">SNR:</span>
                <span class="metric-value {'good' if selected['snr'] >= 2.0 else 'bad'}">
                    {selected['snr']:.2f}
                </span>
            </div>
            
            <div class="metric">
                <span class="metric-label">Dynamic Range:</span>
                <span class="metric-value {'good' if selected['dynamic_range'] >= 0.3 else 'bad'}">
                    {selected['dynamic_range']:.2f}
                </span>
            </div>
            
            <div class="metric">
                <span class="metric-label">Stationarity:</span>
                <span class="metric-value">
                    {selected['stationarity']:.3f}
                </span>
            </div>
            
            <hr>
            
        </div>
    """
    
    # Add backup features
    if len(backups) > 0:
        html_content += """
        <div class="feature-card">
            <h2> Backup Features</h2>
            <p>Alternative targets in case primary feature is not trainable:</p>
        """
        
        for i, backup in enumerate(backups, 1):
            html_content += f"""
            <div style="margin: 15px 0; padding: 10px; background: #ecf0f1; border-radius: 5px;">
                <strong>{i}. {backup['feature_name']}</strong> 
                (Composite: {backup['composite_score']:.3f}, 
                Z-score: {backup['z_score']:.2f}, 
                ICC: {backup['icc']:.3f}, 
                SNR: {backup['snr']:.2f})
            </div>
            """
        
        html_content += "</div>"
    
    # Add visualization
    html_content += f"""
        <div class="feature-card">
            <h2> Visual Analysis</h2>
            <img src="selection_report_v{visit}.png" alt="Feature selection visualizations">
        </div>
    """
    
    # Add top 10 table
    html_content += """
        <div class="feature-card">
            <h2> Top 10 Ranked Features</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Feature Name</th>
                        <th>Type</th>
                        <th>Composite</th>
                        <th>Z-Score</th>
                        <th>ICC</th>
                        <th>SNR</th>
                        <th>Dyn. Range</th>
                        <th>Meets Criteria</th>
                    </tr>
                </thead>
                <tbody>
        """
    
    for _, row in df_ranked.head(10).iterrows():
        meets_icon = "True" if row['meets_criteria'] else "False"
        html_content += f"""
                    <tr>
                        <td>{int(row['rank'])}</td>
                        <td><strong>{row['feature_name']}</strong></td>
                        <td>{row['feature_type']}</td>
                        <td>{row['composite_score']:.3f}</td>
                        <td>{row['z_score']:.2f}</td>
                        <td>{row['icc']:.3f}</td>
                        <td>{row['snr']:.2f}</td>
                        <td>{row['dynamic_range']:.2f}</td>
                        <td>{meets_icon}</td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="feature-card">
            <h2> Interpretation Guide</h2>
            <ul>
                <li><strong>Composite Score:</strong> Weighted combination of all metrics (higher is better)</li>
                <li><strong>Z-Score:</strong> Deviation from normative data (≥1.5 recommended)</li>
                <li><strong>ICC:</strong> Split-half reliability (≥0.6 recommended)</li>
                <li><strong>SNR:</strong> Signal-to-noise ratio (≥2.0 recommended)</li>
                <li><strong>Dynamic Range:</strong> Modulation potential (≥0.3 recommended)</li>
                <li><strong>Stationarity:</strong> Temporal stability (higher is more stable)</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open(output_dir / f"selection_report_v{visit}.html", 'w') as f:
        f.write(html_content)
    
    print(f"Report saved to: {output_dir / f'selection_report_v{visit}.html'}")
    return output_dir / f"selection_report_v{visit}.html"



def track_feature_across_sessions(subject_id, subjects_dir, current_visit):
    """
    Compare selected feature stability across sessions.
    Decide whether to continue with same feature or switch.
    """
    subject_dir = subjects_dir / subject_id
    selection_dir = subject_dir / "nf_selection"
    
    # Load all previous session reports
    session_reports = []
    for v in range(1, current_visit + 1):
        json_path = selection_dir / f"selection_report_v{v}.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                session_reports.append(json.load(f))
    
    if len(session_reports) < 2:
        return None  # Need at least 2 sessions to compare
    
    # Get primary feature from session 1
    primary_feature_name = session_reports[0]['selected_feature']['feature_name']
    
    # Track this feature's metrics across sessions
    tracking_df = []
    for report in session_reports:
        visit_num = report['visit']
        
        # Load full ranking for this visit
        ranking_path = selection_dir / f"feature_ranking_v{visit_num}.csv"
        df_rank = pd.read_csv(ranking_path)
        
        # Find our primary feature
        primary_row = df_rank[df_rank['feature_name'] == primary_feature_name]
        
        if len(primary_row) > 0:
            tracking_df.append({
                'visit': visit_num,
                'feature_name': primary_feature_name,
                **primary_row.iloc[0].to_dict()
            })
    
    df_tracking = pd.DataFrame(tracking_df)
    
    # Compute stability
    icc_stability = df_tracking['icc'].std() < 0.15  # Low variation
    snr_stability = df_tracking['snr'].std() < 1.0
    
    # Decision logic
    current_metrics = df_tracking.iloc[-1]
    
    if current_metrics['meets_criteria'] and (icc_stability or snr_stability):
        decision = "MAINTAIN"
        message = f"Continue training {primary_feature_name} - stable and reliable"
    elif current_metrics['snr'] < 1.5:
        decision = "SWITCH"
        message = f"SNR dropped too low ({current_metrics['snr']:.2f}) - consider switching"
    else:
        decision = "MAINTAIN"
        message = f"Continue with {primary_feature_name} despite some variability"
    
    # Save tracking report
    df_tracking.to_csv(
        selection_dir / f"feature_tracking_through_v{current_visit}.csv",
        index=False
    )
    
    # Create tracking visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Feature Tracking: {primary_feature_name} - Visits 1-{current_visit}", 
                fontsize=14, fontweight='bold')
    
    visits = df_tracking['visit'].values
    
    axes[0, 0].plot(visits, df_tracking['icc'], marker='o', linewidth=2, markersize=8)
    axes[0, 0].axhline(0.6, color='red', linestyle='--', label='Threshold')
    axes[0, 0].set_ylabel('ICC')
    axes[0, 0].set_title('Reliability Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(visits, df_tracking['snr'], marker='o', linewidth=2, markersize=8, color='orange')
    axes[0, 1].axhline(2.0, color='red', linestyle='--', label='Threshold')
    axes[0, 1].set_ylabel('SNR')
    axes[0, 1].set_title('Signal-to-Noise Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].plot(visits, df_tracking['z_score'], marker='o', linewidth=2, markersize=8, color='green')
    axes[1, 0].axhline(1.5, color='red', linestyle='--', label='Threshold')
    axes[1, 0].set_ylabel('Z-Score')
    axes[1, 0].set_xlabel('Visit')
    axes[1, 0].set_title('Deviation from Norm')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(visits, df_tracking['composite_score'], marker='o', linewidth=2, markersize=8, color='purple')
    axes[1, 1].set_ylabel('Composite Score')
    axes[1, 1].set_xlabel('Visit')
    axes[1, 1].set_title('Overall Quality Score')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(selection_dir / f"feature_tracking_v{current_visit}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'decision': decision,
        'message': message,
        'tracking_data': df_tracking,
        'primary_feature': primary_feature_name
    }



if __name__ == "__main__":

    with open("config_master.yml", "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--visit", required=True)
    args = parser.parse_args()

    subject_id = args.subject_id
    visit = int(args.visit)
    subjects_dir = config.get("subjects_dir", "./")
    subjects_dir = Path(subjects_dir)


    report_dict = rank_features(subject_id, subjects_dir, visit)
    generate_selection_report_html(report_dict, subjects_dir)
    
    if int(visit) > 1:
        track_feature_across_sessions(subject_id, subjects_dir, visit)