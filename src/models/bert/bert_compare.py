import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

class ModelComparator:
    def __init__(self):
        """Compare BERT and Baseline model performance"""
        self.baseline_dir = "./artifacts/baseline"
        self.bert_dir = "./artifacts/bert_evaluation"
        
        # Create BERT evaluation directory if it doesn't exist
        os.makedirs(self.bert_dir, exist_ok=True)
        
        print("📊 Model Performance Comparator")
        print("="*50)
    
    def load_baseline_metrics(self):
        """Load baseline model metrics"""
        try:
            with open(f"{self.baseline_dir}/metrics.json", 'r', encoding='utf-8') as f:
                baseline_metrics = json.load(f)
            print("✅ Baseline metrics loaded")
            return baseline_metrics
        except FileNotFoundError:
            print("❌ Baseline metrics not found. Run baseline evaluation first.")
            return None
        except Exception as e:
            print(f"❌ Error loading baseline metrics: {e}")
            return None
    
    def load_bert_metrics(self):
        """Load BERT model metrics"""
        try:
            with open(f"{self.bert_dir}/bert_metrics.json", 'r', encoding='utf-8') as f:
                bert_metrics = json.load(f)
            print("✅ BERT metrics loaded")
            return bert_metrics
        except FileNotFoundError:
            print("❌ BERT metrics not found. Run BERT evaluation first.")
            print("💡 Use: python src/models/bert/bert_evaluate.py")
            return None
        except Exception as e:
            print(f"❌ Error loading BERT metrics: {e}")
            return None
    
    def compare_metrics(self, baseline_metrics, bert_metrics):
        """Compare key metrics between models"""
        
        print("\n🔍 DETAILED PERFORMANCE COMPARISON")
        print("="*60)
        
        # Overall metrics comparison
        print("\n📊 Overall Performance:")
        print(f"{'Metric':<15} {'Baseline':<12} {'BERT':<12} {'Improvement':<12}")
        print("-" * 55)
        
        accuracy_baseline = baseline_metrics.get('accuracy', 0)
        accuracy_bert = bert_metrics.get('accuracy', 0)
        accuracy_improvement = accuracy_bert - accuracy_baseline
        
        f1_baseline = baseline_metrics.get('f1_macro', 0)
        f1_bert = bert_metrics.get('f1_macro', 0)
        f1_improvement = f1_bert - f1_baseline
        
        print(f"{'Accuracy':<15} {accuracy_baseline:<12.4f} {accuracy_bert:<12.4f} {accuracy_improvement:+.4f}")
        print(f"{'F1 (Macro)':<15} {f1_baseline:<12.4f} {f1_bert:<12.4f} {f1_improvement:+.4f}")
        
        # Per-class comparison
        print(f"\n📈 Per-Class F1 Scores:")
        print(f"{'Class':<12} {'Baseline':<12} {'BERT':<12} {'Improvement':<12}")
        print("-" * 52)
        
        classes = ['Negative', 'Neutral', 'Positive']
        class_comparisons = {}
        
        for class_name in classes:
            baseline_f1 = baseline_metrics.get('classification_report', {}).get(class_name, {}).get('f1-score', 0)
            bert_f1 = bert_metrics.get('per_class_metrics', {}).get(class_name, {}).get('f1-score', 0)
            improvement = bert_f1 - baseline_f1
            
            class_comparisons[class_name] = {
                'baseline': baseline_f1,
                'bert': bert_f1,
                'improvement': improvement
            }
            
            print(f"{class_name:<12} {baseline_f1:<12.4f} {bert_f1:<12.4f} {improvement:+.4f}")
        
        return {
            'overall': {
                'accuracy': {'baseline': accuracy_baseline, 'bert': accuracy_bert, 'improvement': accuracy_improvement},
                'f1_macro': {'baseline': f1_baseline, 'bert': f1_bert, 'improvement': f1_improvement}
            },
            'per_class': class_comparisons
        }
    
    def create_comparison_plots(self, comparison_data):
        """Create visualization plots for comparison"""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('BERT vs Baseline Model Comparison', fontsize=16, fontweight='bold')
        
        # 1. Overall metrics comparison
        ax1 = axes[0, 0]
        metrics = ['Accuracy', 'F1 (Macro)']
        baseline_scores = [
            comparison_data['overall']['accuracy']['baseline'],
            comparison_data['overall']['f1_macro']['baseline']
        ]
        bert_scores = [
            comparison_data['overall']['accuracy']['bert'],
            comparison_data['overall']['f1_macro']['bert']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, baseline_scores, width, label='Baseline', alpha=0.8)
        ax1.bar(x + width/2, bert_scores, width, label='BERT', alpha=0.8)
        ax1.set_ylabel('Score')
        ax1.set_title('Overall Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for i, (baseline, bert) in enumerate(zip(baseline_scores, bert_scores)):
            ax1.text(i - width/2, baseline + 0.01, f'{baseline:.3f}', ha='center', va='bottom')
            ax1.text(i + width/2, bert + 0.01, f'{bert:.3f}', ha='center', va='bottom')
        
        # 2. Per-class F1 scores
        ax2 = axes[0, 1]
        classes = list(comparison_data['per_class'].keys())
        baseline_f1 = [comparison_data['per_class'][cls]['baseline'] for cls in classes]
        bert_f1 = [comparison_data['per_class'][cls]['bert'] for cls in classes]
        
        x = np.arange(len(classes))
        ax2.bar(x - width/2, baseline_f1, width, label='Baseline', alpha=0.8)
        ax2.bar(x + width/2, bert_f1, width, label='BERT', alpha=0.8)
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Per-Class F1 Score Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(classes)
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for i, (baseline, bert) in enumerate(zip(baseline_f1, bert_f1)):
            ax2.text(i - width/2, baseline + 0.01, f'{baseline:.3f}', ha='center', va='bottom')
            ax2.text(i + width/2, bert + 0.01, f'{bert:.3f}', ha='center', va='bottom')
        
        # 3. Improvement heatmap
        ax3 = axes[1, 0]
        improvements = []
        improvement_labels = []
        
        # Overall improvements
        improvements.append([comparison_data['overall']['accuracy']['improvement']])
        improvement_labels.append('Accuracy')
        improvements.append([comparison_data['overall']['f1_macro']['improvement']])
        improvement_labels.append('F1 (Macro)')
        
        # Per-class improvements
        for cls in classes:
            improvements.append([comparison_data['per_class'][cls]['improvement']])
            improvement_labels.append(f'F1 ({cls})')
        
        improvements = np.array(improvements)
        
        im = ax3.imshow(improvements, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=0.2)
        ax3.set_yticks(range(len(improvement_labels)))
        ax3.set_yticklabels(improvement_labels)
        ax3.set_xticks([0])
        ax3.set_xticklabels(['BERT - Baseline'])
        ax3.set_title('Performance Improvements')
        
        # Add text annotations
        for i in range(len(improvement_labels)):
            text = ax3.text(0, i, f'{improvements[i][0]:+.3f}', 
                           ha="center", va="center", color="black", fontweight='bold')
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate summary stats
        total_improvements = len([imp for imp in improvements.flatten() if imp > 0])
        avg_improvement = np.mean(improvements.flatten())
        max_improvement = np.max(improvements.flatten())
        min_improvement = np.min(improvements.flatten())
        
        summary_text = f"""
        📊 SUMMARY STATISTICS
        
        ✅ Metrics Improved: {total_improvements}/{len(improvements)}
        📈 Average Improvement: {avg_improvement:+.4f}
        🎯 Best Improvement: {max_improvement:+.4f}
        📉 Worst Change: {min_improvement:+.4f}
        
        🏆 Winner: {'BERT' if avg_improvement > 0 else 'Baseline'}
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=12, va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f"{self.bert_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Comparison plot saved: {self.bert_dir}/model_comparison.png")
    
    def save_comparison_report(self, comparison_data, baseline_metrics, bert_metrics):
        """Save detailed comparison report"""
        
        report = {
            'comparison_summary': {
                'model_versions': {
                    'baseline': 'Traditional ML (TF-IDF + LogisticRegression)',
                    'bert': 'BERT Fine-tuned (dbmdz/bert-base-turkish-cased)'
                },
                'comparison_date': pd.Timestamp.now().isoformat(),
                'overall_winner': 'BERT' if comparison_data['overall']['accuracy']['improvement'] > 0 else 'Baseline'
            },
            'detailed_comparison': comparison_data,
            'raw_metrics': {
                'baseline': baseline_metrics,
                'bert': bert_metrics
            }
        }
        
        # Save JSON report
        with open(f"{self.bert_dir}/comparison_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Create CSV summary
        summary_df = pd.DataFrame({
            'Metric': ['Accuracy', 'F1 (Macro)', 'F1 (Negative)', 'F1 (Neutral)', 'F1 (Positive)'],
            'Baseline': [
                baseline_metrics.get('accuracy', 0),
                baseline_metrics.get('f1_macro', 0),
                baseline_metrics.get('classification_report', {}).get('Negative', {}).get('f1-score', 0),
                baseline_metrics.get('classification_report', {}).get('Neutral', {}).get('f1-score', 0),
                baseline_metrics.get('classification_report', {}).get('Positive', {}).get('f1-score', 0)
            ],
            'BERT': [
                bert_metrics.get('accuracy', 0),
                bert_metrics.get('f1_macro', 0),
                bert_metrics.get('per_class_metrics', {}).get('Negative', {}).get('f1-score', 0),
                bert_metrics.get('per_class_metrics', {}).get('Neutral', {}).get('f1-score', 0),
                bert_metrics.get('per_class_metrics', {}).get('Positive', {}).get('f1-score', 0)
            ]
        })
        
        summary_df['Improvement'] = summary_df['BERT'] - summary_df['Baseline']
        summary_df['Improvement_Pct'] = (summary_df['Improvement'] / summary_df['Baseline']) * 100
        
        summary_df.to_csv(f"{self.bert_dir}/comparison_summary.csv", index=False)
        
        print(f"📄 Comparison report saved: {self.bert_dir}/comparison_report.json")
        print(f"📊 Summary CSV saved: {self.bert_dir}/comparison_summary.csv")

def main():
    """Run model comparison"""
    comparator = ModelComparator()
    
    # Load metrics
    baseline_metrics = comparator.load_baseline_metrics()
    bert_metrics = comparator.load_bert_metrics()
    
    if baseline_metrics is None or bert_metrics is None:
        print("\n❌ Cannot proceed without both metric files.")
        print("Please ensure you have:")
        print("1. Trained and evaluated baseline model")
        print("2. Trained and evaluated BERT model")
        return
    
    # Compare metrics
    comparison_data = comparator.compare_metrics(baseline_metrics, bert_metrics)
    
    # Create visualizations
    print(f"\n📊 Creating comparison visualizations...")
    comparator.create_comparison_plots(comparison_data)
    
    # Save comprehensive report
    print(f"\n📄 Saving comparison report...")
    comparator.save_comparison_report(comparison_data, baseline_metrics, bert_metrics)
    
    # Final summary
    print(f"\n🎉 Comparison completed!")
    print(f"📁 Results saved to: {comparator.bert_dir}")
    
    # Recommendation
    overall_improvement = comparison_data['overall']['accuracy']['improvement']
    if overall_improvement > 0.05:
        print(f"\n🏆 BERT shows significant improvement (+{overall_improvement:.1%})")
        print(f"💡 Recommendation: Use BERT model for production")
    elif overall_improvement > 0:
        print(f"\n📈 BERT shows modest improvement (+{overall_improvement:.1%})")
        print(f"💡 Consider computational cost vs. performance gain")
    else:
        print(f"\n📉 Baseline performs better ({overall_improvement:.1%})")
        print(f"💡 Consider further BERT fine-tuning or data quality")

if __name__ == "__main__":
    main()
