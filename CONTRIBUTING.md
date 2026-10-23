# Contributing

Keep the execution path easy to follow. Use explicit names, small functions with one purpose,
and straightforward control flow. Add a helper only when it removes real duplication or names
an important operation. Comments explain numerical constraints or ownership, rather than
repeating the code. Preserve reference checks and boundary cases when simplifying code.

Keep performance claims tied to the measured model, workload, precision and hardware.
Report failures and incomplete coverage alongside successful results. Put detailed experiment
analysis in `docs/`; keep the README focused on current behavior and the latest verified data.

Before committing, run the relevant CPU tests and inspect the staged diff. GPU changes also
need the existing numerical and performance checks. Documentation changes do not require a
new GPU run.

Use focused commits with a short, descriptive subject. Keep personal emails, hostnames, device
identifiers, credentials and private paths out of commits. Runtime logs stay out of Git.
