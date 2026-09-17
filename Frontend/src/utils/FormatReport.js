// Converts raw report/analysis JSON into clean, readable Markdown —
// used both for on-screen rendering (via react-markdown) and for the
// .md/.pdf download, so what you see is exactly what you get.

export function cleanMarkdown(markdown) {
  return String(markdown ?? "").replace(/<br\s*\/?\s*>/gi, "\n");
}

export function formatReportAsMarkdown(report, threadId) {
  let md = `# Analysis Report\n\n`;
  if (threadId) md += `*Run ID: ${threadId}*\n\n`;
  if (report.problem_type) md += `**Problem type**: ${report.problem_type}\n\n`;
  if (report.target_column) md += `**Target column**: \`${report.target_column}\`\n\n`;

  if (report.dataset_overview) {
    const [rows, cols] = report.dataset_overview.final_shape || [];
    md += `## Dataset Overview\n\n`;
    md += `- **Shape**: ${rows} rows × ${cols} columns\n`;
    md += `- **Columns**: ${(report.dataset_overview.columns || []).map((c) => `\`${c}\``).join(", ")}\n\n`;
  }

  if (report.cleaning) {
    const c = report.cleaning;
    md += `## Cleaning\n\n`;
    md += `- Original rows: ${c.original_rows} → Final rows: ${c.final_rows}\n`;
    md += `- Duplicates removed: ${c.duplicates_removed}\n`;
    md += `- Missing values remaining: ${c.missing_values_remaining}\n\n`;
  }

  if (report.eda?.numerical_summary) {
    md += `## Exploratory Data Analysis\n\n`;
    md += `| Column | Mean | Std | Min | Max |\n|---|---|---|---|---|\n`;
    for (const [col, s] of Object.entries(report.eda.numerical_summary)) {
      md += `| ${col} | ${round(s.mean)} | ${round(s.std)} | ${round(s.min)} | ${round(s.max)} |\n`;
    }
    md += `\n`;

    // Correlations against the target column specifically, if present —
    // the single most decision-relevant number buried in a big matrix.
    const target = report.target_column;
    if (target && report.eda.correlation?.[target]) {
      const corrs = Object.entries(report.eda.correlation[target])
        .filter(([col]) => col !== target)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
      md += `**Correlation with \`${target}\`** (strongest first):\n\n`;
      for (const [col, val] of corrs) {
        md += `- ${col}: ${round(val)}\n`;
      }
      md += `\n`;
    }
  }

  if (report.visualization?.charts?.length) {
    md += `## Visualizations\n\n`;
    md += `${report.visualization.num_charts} chart(s) generated — see Charts tab / attached images.\n\n`;
  }

  if (report.feature_engineering) {
    const fe = report.feature_engineering;
    md += `## Feature Engineering\n\n`;
    md += `- Strategy: ${fe.strategy}\n`;
    md += `- Steps: ${fe.steps_executed} succeeded, ${fe.steps_failed} failed\n\n`;
    if (fe.step_details?.length) {
      for (const step of fe.step_details) {
        md += `- **${step.action}**: ${step.details}\n`;
      }
      md += `\n`;
    }
    if (fe.warnings?.length) {
      md += `**Warnings:**\n`;
      for (const w of fe.warnings) md += `- ⚠️ ${w}\n`;
      md += `\n`;
    }
  }

  if (report.training) {
    const t = report.training;
    md += `## Model Selection\n\n`;
    md += `| Algorithm | CV Score | Status |\n|---|---|---|\n`;
    for (const c of t.candidates_results || []) {
      md += `| ${c.algorithm} | ${round(c.mean_cv_score)} | ${c.status} |\n`;
    }
    md += `\n**Best**: ${t.best_algorithm} (CV score ${round(t.best_mean_cv_score)})\n\n`;
  }

  if (report.hyperparameter_tuning?.best_trial_score !== undefined) {
    const h = report.hyperparameter_tuning;
    md += `## Hyperparameter Tuning\n\n`;
    md += `- Best trial score: ${round(h.best_trial_score)}\n`;
    md += `- Trials completed: ${h.num_trials_completed}\n`;
    md += `- Best params: ${JSON.stringify(h.best_params)}\n\n`;
  }

  if (report.evaluation?.metrics) {
    md += `## Evaluation\n\n`;
    for (const [key, value] of Object.entries(report.evaluation.metrics)) {
      md += `- **${formatLabel(key)}**: ${round(value)}\n`;
    }
    md += `\n`;
  }

  if (report.conclusions) {
    const c = report.conclusions;
    md += `## Conclusion\n\n`;
    md += `- **Best model**: ${c.best_model}\n`;
    md += `- **Final accuracy**: ${round(c.final_accuracy)}\n`;
    md += `- **Final F1**: ${round(c.final_f1)}\n`;
    md += `- **Recommendation**: ${c.recommendation}\n\n`;
  }

  return md;
}

function round(n) {
  return typeof n === "number" ? Number(n.toFixed(3)) : n;
}

function formatLabel(key) {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function formatAnalysisAsMarkdown(summary) {
  let md = `# Dataset Summary\n\n`;

  md += `- **Rows**: ${summary.rows}\n`;
  md += `- **Columns**: ${summary.columns}\n`;
  md += `- **Memory usage**: ${summary.memory_usage}\n`;
  md += `- **Duplicate rows**: ${summary.duplicate_rows}${summary.has_duplicates ? " ⚠️" : ""}\n\n`;

  if (summary.potential_target_columns?.length) {
    md += `## Potential Target Columns\n\n`;
    md += summary.potential_target_columns.map((c) => `- \`${c}\``).join("\n") + "\n\n";
  }

  md += `## Columns\n\n`;
  md += `| Column | Type | Missing |\n|---|---|---|\n`;
  for (const col of summary.column_names) {
    const type = summary.data_types?.[col] ?? "—";
    const missing = summary.missing_values?.[col] ?? 0;
    md += `| ${col} | ${type} | ${missing} |\n`;
  }
  md += `\n`;

  if (summary.numerical_columns?.length) {
    md += `**Numerical**: ${summary.numerical_columns.map((c) => `\`${c}\``).join(", ")}\n\n`;
  }
  if (summary.categorical_columns?.length) {
    md += `**Categorical**: ${summary.categorical_columns.map((c) => `\`${c}\``).join(", ")}\n\n`;
  }

  return md;
}