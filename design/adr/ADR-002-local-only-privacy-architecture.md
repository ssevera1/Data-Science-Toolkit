# ADR-002: Local-Only Privacy-First Architecture

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024 |
| **Decision Makers** | Scott Severance |
| **Category** | Architecture / Security / Privacy |

## Context

Data science and statistics tools often require users to upload sensitive datasets
(medical records, financial data, survey responses with PII). Many existing tools
are cloud-based (Google Colab, JASP Online, Jamovi Cloud) and require data to
leave the user's machine.

The application needed to determine its deployment and data handling model:
should it operate as a cloud service, a hybrid, or a fully local tool?

### Options Considered

| Approach | Pros | Cons |
|---|---|---|
| **Cloud SaaS** | No installation, collaboration, auto-updates | Data leaves machine, privacy compliance issues, hosting costs |
| **Hybrid (local compute, cloud auth)** | Best of both worlds | Still requires network, complex architecture |
| **Fully Local** | Zero privacy risk, no internet required, simple | No collaboration, no auto-updates, user manages install |

## Decision

**Chosen: Fully local architecture** with zero external network dependencies.

## Rationale

1. **Zero data exposure**: User data never leaves the local machine. There are no
   HTTP client libraries imported (no `requests`, no `urllib` for external calls),
   no API keys, no cloud storage integrations. The system is architecturally
   incapable of exfiltrating data.

2. **Regulatory simplicity**: Users working with HIPAA, GDPR, or FERPA-regulated
   data can use the tool without legal review or data processing agreements.
   There is nothing to audit beyond the local machine.

3. **Offline capability**: The application works without internet access after
   initial installation. This is critical for air-gapped environments, field
   research, and locations with unreliable connectivity.

4. **Trust through transparency**: As an MIT-licensed open-source project, users
   can verify the privacy claims by inspecting the source code. The absence of
   network calls is verifiable, not just promised.

## Implementation Details

- **Streamlit config** (`.streamlit/config.toml`):
  - `server.address = "localhost"` - binds only to loopback interface
  - `browser.gatherUsageStats = false` - disables Streamlit's own telemetry
  - `server.enableXsrfProtection = true` - prevents cross-site request forgery

- **Export security** (`core/data_manager.py`):
  - Formula injection prevention: cell values starting with `=`, `+`, `-`, `@`
    are prefixed to prevent CSV injection attacks when files are opened in Excel

- **No external dependencies at runtime**: All computation uses locally installed
  Python packages. No model downloads, no API calls, no telemetry beacons.

## Trade-offs Accepted

- **No collaboration features**: Users cannot share analyses or datasets through
  the application. They must export results and share files manually.

- **No automatic updates**: Users must manually update the application. There is
  no update notification system.

- **No cloud compute**: Heavy computations (large datasets, many models) are
  bounded by the user's local hardware. There is no option to offload to
  cloud GPUs or distributed compute.

- **Single-user sessions**: Each Streamlit session is independent. There is no
  multi-user state, no authentication, no role-based access control.

## Consequences

- The application can be used in any regulatory environment without legal review
- Installation requires Python and pip (acceptable for the target audience)
- Performance is bounded by local hardware
- Users must manage their own backups and version control of datasets
