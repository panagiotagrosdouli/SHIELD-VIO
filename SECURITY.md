# Security and safety policy

SHIELD-VIO is research software for failure-aware visual–inertial estimation. It is not a certified navigation, control, or safety product.

## Supported versions

Security and safety fixes are applied to the current default branch. Historical research revisions and generated artifacts are not maintained as supported releases unless explicitly tagged otherwise.

## Reporting a vulnerability

Please do not open a public issue for:

- a software vulnerability;
- a dependency or supply-chain concern;
- accidental exposure of credentials or private data;
- a failure mode that could create an immediate hazard on a physical platform;
- instructions that could cause users to mistake research logic for a certified safety mechanism.

Use GitHub's private vulnerability-reporting feature when it is available for this repository. If private reporting is unavailable, contact the repository owner privately through the contact method listed on the maintainer's GitHub profile.

Include:

- the affected revision or release;
- the affected file or subsystem;
- reproduction steps or a minimal example;
- expected and observed behavior;
- potential impact;
- any suggested mitigation;
- whether the report involves a physical platform or sensitive data.

## Response expectations

Reports will be acknowledged and triaged according to reproducibility, severity, and potential impact. Research limitations, unsafe assumptions, and misleading claim boundaries may be handled as documentation or design defects even when they are not conventional software vulnerabilities.

## Safety boundary

The navigation shield, failure detector, calibration utilities, and recovery logic are experimental. They must not be treated as independent safeguards for a robot, vehicle, UAV, or other physical system. Real-world deployment requires platform-specific hazard analysis, fault containment, independent supervision, and validation appropriate to the operating environment.
