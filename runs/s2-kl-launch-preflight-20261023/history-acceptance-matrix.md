# Independent historical inventory acceptance matrix

Requirements received: permit the three cryptographically verified historical artifacts without altering original bytes, weakening seed overlap checks, or admitting arbitrary null/unknown/oversized metadata. The frozen override generator establishes conservative board coverage 6600..6649; both replay logs establish 28000..28003. Freeze resolver code and dependencies with registered sources.

Command: `venv\Scripts\python.exe -m unittest test_s2_kl_history -v`

| Requirement | Level / scenario | Observable outcome |
| --- | --- | --- |
| Real full history is usable | Integration: inventory actual repository | No unresolved metadata; full S2-KL manifest passes history check |
| Exact history and evidence | Integration: three original artifacts plus override generator copied into isolated input | Exact artifact and evidence SHA256 values remain inventoried; 6600..6649 and 28000..28003 fully covered |
| Read-only resolution | Integration: compare original and isolated artifact/evidence bytes | Hashes unchanged after inventory |
| No overlap bypass | Contract: synthetic proposed seeds at interior and boundary points of historical spans | `_check_history` rejects with overlap |
| Modified history fails closed | Integration: append valid JSON whitespace to each known artifact | Inventory unresolved and history check rejects |
| Missing/modified evidence fails closed | Integration: missing generator or generator drift | Inventory unresolved and history check rejects |
| Arbitrary null is unknown | Integration: unrelated null-seed metadata | Inventory unresolved and history check rejects |
| Unknown oversized metadata | Integration: valid JSON above 32 MiB | Inventory unresolved and history check rejects |
| Freeze resolver dependency | Contract: inspect executed repository code during resolution and source closure | Every executed local Python module is included by SHA256 in source closure |

The tests do not depend on a resolver module name or implementation algorithm. Existing scanner and history-check entry points are exercised directly. Local executed dependencies are discovered by tracing repository Python frames during a successful fixture inventory, avoiding an implementation-shaped module list. Registered-source drift rejection is already covered by the existing immutable registration acceptance suite.

## Recorded RED baseline

Executed before production changes: `venv\Scripts\python.exe -m unittest test_s2_kl_history -v`.

Result: **9 tests, 5 expected failures, 4 passes; 33.629 seconds**. Full output: `history-acceptance-red.log`. Failures are missing historical resolution, including the real full inventory and isolated verified fixtures. Both inventories report exactly 72 unresolved entries: 70 null-seed override rows plus two oversized replay artifacts. Fixture raw hashes were verified successfully before tests executed. No syntax, import, dependency, or setup errors occurred. The existing fail-closed guards pass. The complete suite is now frozen for implementation.
