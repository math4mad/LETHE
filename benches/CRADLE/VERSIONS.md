# Birth Certificate — Version Ledger · 出生证版本台账

**Law (owner’s standing order, 2026-09-16):** 出生证若改，原始历史版本必须保留，不得覆盖。
Every version of the birth certificate is an immortal artifact: revisions are
**born beside** their predecessors — never over them. The working copy is a
convenience; `archive/` is the truth. Each sealed version:

- is copied to `archive/birth-certificate-v<X.Y>-<date>.pdf` **before** the working
  copy is recompiled (first the seal, then the forge);
- receives its own `bin/pin.sh` receipt naming the source tex sha it was cast from;
- is entered below, newest first — an append-only table; a row is never edited,
  only succeeded;
- and the gate mirror `website/birth-certificate.pdf` is updated **after** the
  archive row exists, with `bin/pin.sh --verify` proving which working byte it copies.

## The versions · 诸版

| v | sealed (UTC) | certificate sha256 (16) | cast from tex sha | hand · tool | note |
|---|---|---|---|---|---|
| 1.0 | 2026-09-16T01:57Z · archived 02:22Z | `180e301aca164c07…` | `f193107627febe52…` | lethe · tectonic 0.17.0 | **初版** — first compile of the received tex; 4 pp; archived under the append-only law the same morning it was born, byte-identical by construction |

*Future rows append here. The ledger of certificates is itself a certificate: what is written does not change; what changes, is written anew.*
