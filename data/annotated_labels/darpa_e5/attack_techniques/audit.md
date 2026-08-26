# DARPA TC E5 entity-level ATT&CK labels with exact event provenance

Only exact CDM event + PIDSMaker anchor + TA5.1 report joins are final.
Incomplete attack scenes remain explicitly unannotated.

| Dataset | Attack | Raw event UUID | Graph endpoint ID | Anchor / role | Raw anchor events | ATT&CK | Report |
|---|---|---|---|---|---:|---|---|
| trace5 | trace_firefox_drakon_0514 | `89E6049E-9653-C1C2-B0AD-DB11AACB3BC4` | `89E6049E-9653-C1C2-B0AD-DB11AACB3BC4:predicateObject` | `2B7AB6A4-71F3-BE5F-E654-B797622837B3` / object | 39 (selected UUID=1) | T1071.001 / Command and Control | §7.3.4, PDF p.61,63,64 |
| theia5 | theia_firefox_drakon_0515 | `37408DB1-CDB1-5FC0-BD5A-E60FD15BC192` | `37408DB1-CDB1-5FC0-BD5A-E60FD15BC192:predicateObject` | `80370C6E-859F-D0CB-142A-500000000000` / object | 55 (selected UUID=1) | T1071.001 / Command and Control | §8.6, PDF p.79 |
| cadets5 | cadets_nginx_drakon_0517 | `8DCFA717-003C-4641-900F-9562F86994E8` | `8DCFA717-003C-4641-900F-9562F86994E8:predicateObject2` | `98A2F9DC-78AF-11E9-B41B-D4AE52C1DBD3` / object2 | 3 (selected UUID=1) | T1190 / Initial Access | §10.4, PDF p.100,102 |
| cadets5 | cadets_nginx_drakon_0517 | `2A0E7E41-B435-4BDD-98C9-C0AF9D3FB81F` | `2A0E7E41-B435-4BDD-98C9-C0AF9D3FB81F:predicateObject` | `98A5B5B4-78AF-11E9-B41B-D4AE52C1DBD3` / object | 149 (selected UUID=1) | T1105 / Command and Control | §10.4.3.2, PDF p.102 |
| cadets5 | cadets_nginx_drakon_0517 | `CCB63A8F-F44F-497C-9A56-2EE6ABFB3DDC` | `CCB63A8F-F44F-497C-9A56-2EE6ABFB3DDC:predicateObject` | `1E2E3C6D-77E5-11E9-A28B-D4AE52C1DBD3` / actor | 574 (selected UUID=1) | T1071.001 / Command and Control | §10.4.3.3, PDF p.103 |
| clearscope5 | clearscope_tester_0517 | `2C0E135D-0EC9-9B86-67E1-8011D37CF09B` | `2C0E135D-0EC9-9B86-67E1-8011D37CF09B:predicateObject` | `00000000-0000-0000-0000-000001BEEBCA` / actor | 67 (selected UUID=1) | T1095 / Command and Control | §10.12.5, PDF p.128,129 |

## Scene coverage

| Platform | Attack | Status | Exact rows |
|---|---|---:|---:|
| theia5 | theia_firefox_drakon_0515 | exact | 1 |
| cadets5 | cadets_nginx_drakon_0516 | incomplete | 0 |
| cadets5 | cadets_nginx_drakon_0517 | exact | 3 |
| trace5 | trace_firefox_drakon_0514 | exact | 1 |
| clearscope5 | clearscope_appstarter_0515 | incomplete | 0 |
| clearscope5 | clearscope_firefox_0517 | incomplete | 0 |
| clearscope5 | clearscope_lockwatch_0517 | incomplete | 0 |
| clearscope5 | clearscope_tester_0517 | exact | 1 |
