// Converts raw report/analysis JSON into clean, readable Markdown —
// used both for on-screen rendering (via react-markdown) and for the
// .md/.pdf download, so what you see is exactly what you get.

export function formatReportAsMarkdown(report, threadId) {
  let md = `# Report\n\n`;
  if (threadId) md += `*Run ID: ${threadId}*\n\n`;

  if (report.evaluation) {
    md += `## Evaluation\n\n`;
    for (const [key, value] of Object.entries(report.evaluation)) {
      if (typeof value === "object" && value !== null) continue; // skip nested artifact paths
      md += `- **${formatLabel(key)}**: ${formatValue(value)}\n`;
    }
    md += `\n`;
  }

  if (report.eda?.numerical_summary) {
    md += `## Numerical Summary\n\n`;
    for (const [col, stats] of Object.entries(report.eda.numerical_summary)) {
      md += `**${col}**\n`;
      for (const [stat, val] of Object.entries(stats)) {
        md += `- ${formatLabel(stat)}: ${formatValue(val)}\n`;
      }
      md += `\n`;
    }
  }

  return md;
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

function formatLabel(key) {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatValue(value) {
  if (value === null || value === undefined) return "N/A";
  if (typeof value === "number") return Number.isInteger(value) ? value : value.toFixed(3);
  return String(value);
}