"""Build reviewed, source-linked OpTC ATT&CK annotations.

Input rows must already join an optc-labels analyst task to one raw eCAR event
and at least one PIDSMaker malicious actor/object UUID.  The review table below
is intentionally keyed by both task ID and event ID; it cannot annotate a
different event merely because its prose looks similar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OPTC_LABELS_REPOSITORY = "https://github.com/AT03380/optc-labels.git"
OPTC_LABELS_COMMIT = "64c9f9b2e1a15bf3c2789d89d93dc0724cb0d4fa"
TASKS_ZIP_SHA256 = "b8ddd6d2d82ebbdc637f158cb7e7537c808ddd6968079653405844e1efc83c77"
MALICIOUS_ZIP_SHA256 = "9feb73e29c07fd41bbe5670c111bd396bc101991beb39ca3721ddf6565650c2a"

# task_id: (raw_event_id, reviewed primary parent/sub-technique, reference tactic)
REVIEWED = {
    "f0a266d9-a7b2-4c6e-b15b-7719099f5294": ("5cedea14-aed8-4ca4-98e8-642c816ed754", "T1548.002", "Privilege Escalation"),
    "ba87acb4-7cd7-4f36-87e4-4c07ef711b0a": ("cd005a16-9176-4fb9-853d-38ac429147d6", "T1003", "Credential Access"),
    "a5324b01-decb-409d-b6d9-ae49014608a4": ("9cc75983-4433-4ef1-b734-5a48a2346b39", "T1055", "Defense Evasion"),
    "59a13e24-46c2-4ccf-b740-452b465114de": ("3ee3e6e9-0601-46a4-90fa-f823c68b463c", "T1112", "Persistence"),
    "7c89d575-7c5e-4023-b14b-6198545ae12e": ("944a93d0-0c7c-4590-b48e-d081d36269b9", "T1057", "Discovery"),
    "8938eb66-3a5a-4f89-aace-0f2197a4ed87": ("b0331e2c-2285-45fb-9644-a443700edd02", "T1055", "Defense Evasion"),
    "dcc04199-7256-4f16-beef-b290c57b1586": ("989d6566-697b-4025-aca5-5ad344c1ef96", "T1113", "Collection"),
    "08a88f97-36b4-45d4-867e-e0eb04a635cf": ("2e617f24-d5cd-45f2-8130-73673c7073b6", "T1018", "Discovery"),
    "87567332-306d-418d-86e1-0f8bad0aff4f": ("416f0e20-de02-4f63-97d7-928399722e5b", "T1018", "Discovery"),
    "7a1923d1-adf1-4bb3-badb-f6b6c1514433": ("9125f70e-c7d2-46fe-b04b-1e6cb1bc7d49", "T1018", "Discovery"),
    "cd84f337-6e95-4f3c-961c-be65327c2ed7": ("f7cfcf76-a672-47cd-8063-8ebc4f8f7f4d", "T1018", "Discovery"),
    "a9365bd7-fff1-40aa-bde6-46bfe9bf3ad2": ("8acf2c27-cd2d-4cf1-a064-270bb423c8d7", "T1204.002", "Execution"),
    "858a2b01-eb11-46bc-b977-d45f22415dc3": ("58677aed-426c-4b12-bc99-aa4a858797ae", "T1059.001", "Execution"),
    "77dc4049-a146-4bf7-8a55-f95201a69309": ("0fa4352d-2bed-4ed0-a06e-4daff59b2464", "T1069.002", "Discovery"),
    "733df162-1d6f-4e44-946f-fedd01c1e0f8": ("71b0e667-f9a7-4ba3-aeee-54ec96e9d30f", "T1018", "Discovery"),
    "f3920000-bc1b-4f0e-825b-29e462252af5": ("24220a6b-9129-43d3-944e-8b56143fbc0a", "T1548.002", "Privilege Escalation"),
    "14efdd55-328f-447e-968b-c1a7a7dd4660": ("49530707-6416-4ff3-a911-3bdca783c24a", "T1552.006", "Credential Access"),
    "f5cade9c-8b4d-48c3-81e3-66dd7b4cb4b0": ("fab35a85-bebf-4f5c-a58b-7a0c370c1eb8", "T1548.002", "Privilege Escalation"),
    "3c4809a5-9906-4431-aae3-29b4160bef19": ("65c5b5c1-a49e-447a-80a8-70ecc34814eb", "T1548.002", "Privilege Escalation"),
    "23ede4ca-ca0b-43e7-bc93-ccd163f14dfa": ("ad9600b0-d610-4b7a-b982-cc76a0150cfa", "T1047", "Execution"),
    "ebb6973c-7c17-4ad8-90fb-b8a1a4d02dd6": ("7af7404b-1627-46cc-9bda-cc2ac592b153", "T1546.003", "Persistence"),
    "87eb878a-a3c0-47fa-975e-546920193215": ("c359ccad-b6be-4c00-80cf-38b95b3b5b1a", "T1012", "Discovery"),
    "a33b87fe-7065-4c03-a1de-208524629b71": ("8aa13498-f515-4d8d-bec7-a97ea2928de1", "T1105", "Command and Control"),
    "7b4aee5c-06eb-41e8-9e21-617f3bf330c7": ("d1048c30-a0f3-4afa-95d8-3d5e33cdeac9", "T1572", "Command and Control"),
    # The official log says export.zip was exfiltrated with nc.exe on a separate
    # port; T1048 describes the action, while RDP is only session context.
    "dce5211a-21d9-493e-aad3-b915e7f92346": ("a1f31ea0-e50c-4ddb-8aac-9317914bd27a", "T1048", "Exfiltration"),
    "612bbc3a-8a9b-4b2e-94b3-a1d94389af3a": ("7fa3e75b-196a-47d4-9f2a-65f0e61de408", "T1059.003", "Execution"),
    "ecbd3437-8542-4732-9e1c-6351eed84f44": ("6828b8c3-bbcb-4ecd-9e2a-abe50951b50a", "T1055", "Defense Evasion"),
}

# task_id: (official task-record hash, exact raw-event hash, exact PIDSMaker
# anchor-record hash).  These selected-record pins prevent a self-consistent
# rewrite of evidence wrappers from silently changing the reviewed source row.
REVIEWED_SOURCE_PINS = {
    "08a88f97-36b4-45d4-867e-e0eb04a635cf": ("3830f2a26b7d50d3c40105ebb8f59f1eff3d8f8d771c2e9f29632e8d4e3f56b8", "30912dc9705bbaf19efe9e89dbc30695b50a00db0cbbbd32d3e555a7330d819c", "6e4fd071df9600195a34e6e875ff8104c8d91d07cb56fe53c35733789ec443bd"),
    "14efdd55-328f-447e-968b-c1a7a7dd4660": ("600411f467493392fb794710e730586df61b9be38af287b2d142391aaba27a80", "1f108f86eeb5fad572535f87376dab113f2839b4015aaf132f604fb0a77ca74f", "9505c92a4d7bf1f77779c9eeb9dfc47b1673cc3ecf117ee6cfc6af78e12ec7bb"),
    "23ede4ca-ca0b-43e7-bc93-ccd163f14dfa": ("b136bdbb3c4bb5633d679f750981bb73ff1514573751ac6f4d188f9bb93df6f6", "31b570ddcbc6423aed947210cafa970779959ed3831178b533d6456a512cfcfc", "22d656ed5516890607e9088419ad91a4b2d674280a10659138fd089f1a1f4d14"),
    "3c4809a5-9906-4431-aae3-29b4160bef19": ("3359eccc8a9b334ffb525b2a98e8119e45bea61fb75a25a324f2123363f5e4e8", "f6daa7b8307868e85412ebb619e131597fdde79cf32b304ca3a6510613a18e04", "9505c92a4d7bf1f77779c9eeb9dfc47b1673cc3ecf117ee6cfc6af78e12ec7bb"),
    "59a13e24-46c2-4ccf-b740-452b465114de": ("1f4f0e4cfb1989108305247cb3a2338c93a077c3f6f78edfcc54360f46fb7822", "b7e644224cb65bfe7bd8521b048eddd9c3284cd7e4b1109c666219c5b0361c21", "6e4fd071df9600195a34e6e875ff8104c8d91d07cb56fe53c35733789ec443bd"),
    "612bbc3a-8a9b-4b2e-94b3-a1d94389af3a": ("2fcde4266ea4295a276ad448d5e9c19a535c51d527ef5bfb030cfcbe8b3fce1b", "a4d4797b39a46924d06c48f43eb386d65654628e93fd3c44feedbbd35ee26f65", "8d9f1c0d9fd91b92359f5be7a865d75eed32542e5c4007f469436b73b8b0595c"),
    "733df162-1d6f-4e44-946f-fedd01c1e0f8": ("7dabc8e5a963add1d0e76138359ac351d14a4b2679c9f38fc37e058e5b42d874", "93b43f22107a5bfeb8952427c5482415d995d7be537204315e9a75fc45455634", "9505c92a4d7bf1f77779c9eeb9dfc47b1673cc3ecf117ee6cfc6af78e12ec7bb"),
    "77dc4049-a146-4bf7-8a55-f95201a69309": ("db183ffb38b63a683cc6b78054212dc5ee4bb6cde32a8fe789715cd26f4c53f7", "711922e9694b38c9650d908cb879be2e77a9e131568cae6e91768e1a2b8315b8", "9505c92a4d7bf1f77779c9eeb9dfc47b1673cc3ecf117ee6cfc6af78e12ec7bb"),
    "7a1923d1-adf1-4bb3-badb-f6b6c1514433": ("037b39637ecdb39606e8e8aa844d78bf0c4482bab7de85c0bfd168afdd79e93a", "9a7d3a06c7cd4e4d5930cf04e8866459f7bcfbbb352f7a08804d1f1b392a9a0a", "6e4fd071df9600195a34e6e875ff8104c8d91d07cb56fe53c35733789ec443bd"),
    "7b4aee5c-06eb-41e8-9e21-617f3bf330c7": ("f3044364d565a20895bb284539c159a363f492d435c7e77d86ba7136413b3318", "ecf888088cfa33d0cb10db003dca998689b6974272166e6195b9b788198cd1b4", "f8233841b09a580d059484006cd94dbdf50ba22d947ccb902d2040e4eec10c89"),
    "7c89d575-7c5e-4023-b14b-6198545ae12e": ("a4a0b1c930418181faa974d41f097ac9cd4e3cfb4d2697b5ba2a9e8ef3d39a7a", "c4088262620b7433a317c9c0668b77d446b25c601747eebdaebba77f283884c7", "6e4fd071df9600195a34e6e875ff8104c8d91d07cb56fe53c35733789ec443bd"),
    "858a2b01-eb11-46bc-b977-d45f22415dc3": ("91d61089036045b48e7bf0384a3d5c7cf7237971d80d0be84ba9b6e921dab118", "468ea21f72da582cbae5b9092f1d764952eb2fa504c56b708718f65278e26d4b", "9505c92a4d7bf1f77779c9eeb9dfc47b1673cc3ecf117ee6cfc6af78e12ec7bb"),
    "87567332-306d-418d-86e1-0f8bad0aff4f": ("7451a65291a6025703833b169abc473e631bb20e4793c3de125b191022f844a0", "232341cb1fc319349c738c5709df29e944582ffdf10f442ea85e43f5d124089b", "6e4fd071df9600195a34e6e875ff8104c8d91d07cb56fe53c35733789ec443bd"),
    "87eb878a-a3c0-47fa-975e-546920193215": ("ad8d4b3225a6fa8eb04d9e1908ed6e1b9ed0f2221ea3cc29b260d4e6059ad193", "ffefac206f704083d99069e7116ee759b7a2103f9f091fc07fb61826d4301d60", "f8233841b09a580d059484006cd94dbdf50ba22d947ccb902d2040e4eec10c89"),
    "8938eb66-3a5a-4f89-aace-0f2197a4ed87": ("4ab69207d69ce4e38cd48f854b48f449f0cef1fefa829c1a8fcac8274b6c8b19", "72dc230b0a835cfda14ee3332a5f5772489de9e2f478e00c2be44c1c90c20614", "6e4fd071df9600195a34e6e875ff8104c8d91d07cb56fe53c35733789ec443bd"),
    "a33b87fe-7065-4c03-a1de-208524629b71": ("e2ebfa1a458c683257b5d2ea4db7b225a6ec1a2833862d436fa04e60c077d6a2", "3afe47fc889675f4133f9d2804413936aabbf8f8cc5514ccb768bb2eb0660b0d", "f8233841b09a580d059484006cd94dbdf50ba22d947ccb902d2040e4eec10c89"),
    "a5324b01-decb-409d-b6d9-ae49014608a4": ("10637dc0caa036f317663dcd608206cce881da7a810d46a470c0bca261541ddb", "1dea3af5a9ffb702ebd3c4e62792cc6a4b792d815869548e43377dbfded29fd6", "6e4fd071df9600195a34e6e875ff8104c8d91d07cb56fe53c35733789ec443bd"),
    "a9365bd7-fff1-40aa-bde6-46bfe9bf3ad2": ("25a9c1b17e04528a31f2b8624230a755fb2d6fc82ec09cd0003ef8bc22ada915", "8de8169200c2db072f5626be3c5be21cc33f0e12fc741b73070a7cd3f84c5798", "36e1e99c58bee4051f10d926cdcf9dca173f4af72061c99e3f6225d6fea8d92e"),
    "ba87acb4-7cd7-4f36-87e4-4c07ef711b0a": ("b2e991f5c0dc28838291d077290c7de1cabeaec74319616e5cc5949643e01d82", "e23e61acae1d6c30be43c5279e103cf689621b4072fc3eaabc33e5ca6197bfc3", "6e4fd071df9600195a34e6e875ff8104c8d91d07cb56fe53c35733789ec443bd"),
    "cd84f337-6e95-4f3c-961c-be65327c2ed7": ("48ae9e058ca38595f5b72f503b03b774d5dd0a7e19890d645cbf0729564795bf", "cee8f53fb1fdae3c44eab7507042fbf2d4fa7866afcbe92ab69ca68238d46ffd", "6e4fd071df9600195a34e6e875ff8104c8d91d07cb56fe53c35733789ec443bd"),
    "dcc04199-7256-4f16-beef-b290c57b1586": ("532c77e9d0a4dc2b04521510ecaafdcf2cc91f5eaf157658e265fb7a3598e1f8", "1c25bc26474ad535fcd533342733cfb75ba1b2909ea449bc42719ceef28c6b22", "6e4fd071df9600195a34e6e875ff8104c8d91d07cb56fe53c35733789ec443bd"),
    "dce5211a-21d9-493e-aad3-b915e7f92346": ("475ec7d648f1bb099f1407080841b49e169b505c57bc20ebcba2659e0bd23a72", "8445f02c414cdd78b845fa336911dabf8e7acd76add69f13c105661193a0d55d", "6b1a21ba69e394f4f43a723170e829e7513a9986571392f9c229bc4f0cc31cab"),
    "ebb6973c-7c17-4ad8-90fb-b8a1a4d02dd6": ("7f450bfc3b1448805ba96a433bab7f608fab5a73fdbe1a64fa580a4948aab507", "5716f9a7fdd31227fa0912e5b70fe74b734ed8af722603aedab7b22ddcaf53de", "f8233841b09a580d059484006cd94dbdf50ba22d947ccb902d2040e4eec10c89"),
    "ecbd3437-8542-4732-9e1c-6351eed84f44": ("777bb3c39bb898f846ba9c4bb884e3f0fa25d28ed352b7f203e252e61b382bf8", "34a6e3ea2378ecb73b6b1b788cbb14f2d7e2c57d3ce981c9a7b5605269214eb4", "8d9f1c0d9fd91b92359f5be7a865d75eed32542e5c4007f469436b73b8b0595c"),
    "f0a266d9-a7b2-4c6e-b15b-7719099f5294": ("031d203661084bd49ef81f34f6d53bed385975fd6ba7b636c49e57d86c4d131d", "fdb9901a065c9d1994e0567894ad782c32cf4828f311fab2e840c1ce0b69932f", "7b9840db86337e5017d5d1ec7ce0048567e9e00e790449ecb1b3d5ba28bf46e9"),
    "f3920000-bc1b-4f0e-825b-29e462252af5": ("8f82d7c1509cc95d012eccfcac71c248f17f9e054f7442687452ac0a33ac26e3", "6ee6b5d55f263cdd0dc3ccb8bdb7c1cf0aa91cf8c50ac3df573e2a925c2fe5b7", "9505c92a4d7bf1f77779c9eeb9dfc47b1673cc3ecf117ee6cfc6af78e12ec7bb"),
    "f5cade9c-8b4d-48c3-81e3-66dd7b4cb4b0": ("5980fa50102e2b32ebbe2fcdc8bd1559b38bb0ec2f1dc4044ee33b5718573131", "42b25f42c3f660014e5c01b71fbbc29484761b1cee465702925f5034728f6f1e", "9505c92a4d7bf1f77779c9eeb9dfc47b1673cc3ecf117ee6cfc6af78e12ec7bb"),
}

# task_id: canonical hash of the complete, path-sanitized evidence record.
# Unlike the three selected-row hashes above, these pins also bind the official
# log text and every raw-event/anchor field used by the manual review.
REVIEWED_EVIDENCE_PINS = {
    "08a88f97-36b4-45d4-867e-e0eb04a635cf": "b2251f2477815d898988b1fc318f73debcb73ea64b253368b3a2abddd7823800",
    "14efdd55-328f-447e-968b-c1a7a7dd4660": "e3972e620daf8fb9c6bf47d9f315051ae79f7f73d0f77d5bdbf64ad12c51e574",
    "23ede4ca-ca0b-43e7-bc93-ccd163f14dfa": "acce03e2227bc2a7bbba026142293d802891962f31842b1577d9a89119f7c626",
    "3c4809a5-9906-4431-aae3-29b4160bef19": "24ae60d6b272b79e6f9ac1398290df0ed1b7b50292a458221c948fe5cdc5d67e",
    "59a13e24-46c2-4ccf-b740-452b465114de": "813ae7beb7c71db3c8a75f826bdf44787fcebd0ee1f6af893844413fc7733c15",
    "612bbc3a-8a9b-4b2e-94b3-a1d94389af3a": "b792d71705c545bb664dc2b10b87e2871d8bcc3c59492901725953e6352312be",
    "733df162-1d6f-4e44-946f-fedd01c1e0f8": "1317bde02a3c01565848e3dc39d50e380df78ab3732e5a7bc07565ff60a7e7da",
    "77dc4049-a146-4bf7-8a55-f95201a69309": "e9199b1bf185fad188db43bc9df2d75ff005a94022157c4447f30e3356ba21e9",
    "7a1923d1-adf1-4bb3-badb-f6b6c1514433": "38f36539840cacb19b53b87937d542f41a88b6f99b48e09b6016744ddc29dbd4",
    "7b4aee5c-06eb-41e8-9e21-617f3bf330c7": "c4223fc81b18ecd94bada9a3ea085eb922c97f1dd370d02b796661a2c46c68f9",
    "7c89d575-7c5e-4023-b14b-6198545ae12e": "7889a8e07dbeae63a4b10d34239675d96871c3294390d0955931006a6993a20f",
    "858a2b01-eb11-46bc-b977-d45f22415dc3": "dcbc4322c53473a886be07df3f9613bc11b08f694c264d4f197cf8feca051d25",
    "87567332-306d-418d-86e1-0f8bad0aff4f": "616bc204c4951214f921fc5b7f97b939acd44462b1c5c0f790eaaeffc59e6348",
    "87eb878a-a3c0-47fa-975e-546920193215": "1133dce79ecb9c3388c1777ae7f345345d71ee780a956a51e7b238fb60e59879",
    "8938eb66-3a5a-4f89-aace-0f2197a4ed87": "5c77ab35fd70586cdd59bcdb778a6be2a29818334e30ae5b6e842f40461d1859",
    "a33b87fe-7065-4c03-a1de-208524629b71": "96293e1859ad416ed1a6e21583dad7f80607198d570dd78d6cb4195a124bc583",
    "a5324b01-decb-409d-b6d9-ae49014608a4": "9004af327886a334e52c705c9a1d3fbe4c3230fc64ce879d2e191159247f47b9",
    "a9365bd7-fff1-40aa-bde6-46bfe9bf3ad2": "071c8ddff90bc24049b53d380cf08ba0414f5217251be4a3064a506264866668",
    "ba87acb4-7cd7-4f36-87e4-4c07ef711b0a": "2739c3dba7390a2c3ddf3dc8abd645ce6a21816552420507232109446342af97",
    "cd84f337-6e95-4f3c-961c-be65327c2ed7": "56da5804934a85ab4949ec8827fb1145f6cba600c6cfa1908584eb41e2685a6b",
    "dcc04199-7256-4f16-beef-b290c57b1586": "33b99c7b4965e3a1405cf0a5d7843338f196f30cbad71f85643e84d08b428abf",
    "dce5211a-21d9-493e-aad3-b915e7f92346": "22939f573e46669263c94475a148d58601de569d41b602574928f6174fcb9818",
    "ebb6973c-7c17-4ad8-90fb-b8a1a4d02dd6": "8ed572df1335c9f2f6cd07f2a4c1ee8dfc9998fb66b4d1140e126d23be5b2f8e",
    "ecbd3437-8542-4732-9e1c-6351eed84f44": "6fa919dd42c3b333c3342bc32c43e7294159418acb6f43fbda273bb2c3d49e54",
    "f0a266d9-a7b2-4c6e-b15b-7719099f5294": "eca4d2b24c94f8dfbbbb5efc89ac552404861224afa8ec515859d0dbc3bae6b9",
    "f3920000-bc1b-4f0e-825b-29e462252af5": "a9dacfb9acec29ff95de6e519d6b873fc646ae843822f21c7d2ffe50c811fb43",
    "f5cade9c-8b4d-48c3-81e3-66dd7b4cb4b0": "20e1cc5b81521bc321bafc31d8e3c16b8f898bf40aae497e8b41b630c6ea9b14",
}

HOSTS = {"optc_h051": "H051", "optc_h201": "H201", "optc_h501": "H501"}
SCENES = {"optc_h201": "day1", "optc_h501": "day2", "optc_h051": "day3"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def _sanitize_anchor(anchor: dict) -> dict:
    source_path = str(anchor.get("pidsmaker_path") or "")
    marker = "Ground_Truth/"
    if marker in source_path:
        source_path = marker + source_path.split(marker, 1)[1]
    return {
        "source_path": source_path,
        "source_file_sha256": anchor.get("pidsmaker_file_sha256"),
        "source_row": anchor.get("line"),
        "source_record_sha256": anchor.get("record_sha256"),
        "node_kind": anchor.get("node_kind"),
        "node_description": anchor.get("node_description"),
        "pidsmaker_export_index": int(anchor.get("pidsmaker_legacy_index")),
    }


def build(candidates_path: Path, output_dir: Path) -> dict:
    tactic_path = REPO_ROOT / "data/attack_knowledge/mitre_attack/technique_to_tactic.json"
    tactic_map = json.loads(tactic_path.read_text())
    input_rows = [json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_task = {str(row.get("task_id")): row for row in input_rows}
    if len(by_task) != len(input_rows):
        raise ValueError("candidate input contains duplicate task IDs")
    if set(REVIEWED_SOURCE_PINS) != set(REVIEWED_EVIDENCE_PINS) or set(REVIEWED) != set(REVIEWED_SOURCE_PINS):
        raise RuntimeError("reviewed OpTC event, source, and evidence pin sets differ")
    source_records, draft_annotations = [], []
    for task_id, (expected_event, technique, tactic) in REVIEWED.items():
        row = by_task.get(task_id)
        if row is None:
            raise ValueError(f"reviewed task {task_id} is absent from candidate input")
        raw = row.get("raw_event") or {}
        if str(raw.get("event_id")) != expected_event:
            raise ValueError(f"reviewed task {task_id} no longer resolves to {expected_event}")
        if str((row.get("task_source") or {}).get("zip_sha256")) != TASKS_ZIP_SHA256:
            raise ValueError(f"reviewed task {task_id} has a different tasks archive")
        if str((raw.get("source") or {}).get("zip_sha256")) != MALICIOUS_ZIP_SHA256:
            raise ValueError(f"reviewed task {task_id} has a different raw-event archive")
        matching = raw.get("matching_pids_anchors") or []
        selected = next((item for item in matching if item.get("role") == "actor"), None)
        selected = selected or next((item for item in matching if item.get("role") == "object"), None)
        if not selected or len(selected.get("anchors") or []) != 1:
            raise ValueError(f"reviewed task {task_id} lacks one exact PIDSMaker anchor")
        role = str(selected["role"])
        anchor_uuid = str(raw[f"{role}ID"])
        anchor_evidence = _sanitize_anchor(selected["anchors"][0])
        observed_pin = (
            str((row.get("task_source") or {}).get("task_record_sha256") or ""),
            str((raw.get("source") or {}).get("record_sha256") or ""),
            str(anchor_evidence.get("source_record_sha256") or ""),
        )
        if observed_pin != REVIEWED_SOURCE_PINS[task_id]:
            raise ValueError(f"reviewed task/raw/anchor source row changed for {task_id}")
        if tactic not in tactic_map.get(technique, []):
            raise ValueError(f"{technique} is not mapped to {tactic} in the released ATT&CK registry")
        source_record = {
            "record_type": "optc_manual_annotation_evidence",
            "task_id": task_id,
            "official_log_text": row.get("official_log_text"),
            "task_record_sha256": (row.get("task_source") or {}).get("task_record_sha256"),
            "task_source": {
                "repository": OPTC_LABELS_REPOSITORY,
                "commit": OPTC_LABELS_COMMIT,
                "archive": "tasks/tasks.zip",
                "archive_sha256": TASKS_ZIP_SHA256,
                "entry": "tasks.json",
                "task_index": (row.get("task_source") or {}).get("task_index"),
            },
            "raw_event": {
                "event_id": expected_event,
                "timestamp": row.get("timestamp"),
                "hostname": row.get("hostname"),
                "actorID": raw.get("actorID"),
                "objectID": raw.get("objectID"),
                "action": raw.get("action"),
                "object": raw.get("object"),
                "relevant_properties": raw.get("relevant_properties") or {},
                "source_record_sha256": (raw.get("source") or {}).get("record_sha256"),
                "source_record_number": (raw.get("source") or {}).get("line"),
                "archive": "labels/malicious.zip",
                "archive_sha256": MALICIOUS_ZIP_SHA256,
                "entry": "malicious.json",
            },
            "anchor": {"role": role, "node_uuid": anchor_uuid, **anchor_evidence},
        }
        source_record["record_sha256"] = _canonical_hash(source_record)
        if source_record["record_sha256"] != REVIEWED_EVIDENCE_PINS[task_id]:
            raise ValueError(f"reviewed complete evidence record changed for {task_id}")
        source_records.append(source_record)
        draft_annotations.append({
            "record_type": "source_linked_attack_annotation",
            "annotation_id": f"optc-{task_id}",
            "dataset": "optcday1",
            "scene": SCENES[str(row.get("dataset"))],
            "source_partition": row.get("dataset"),
            "host_id": HOSTS[str(row.get("dataset"))],
            "anchor_uuid": anchor_uuid,
            "anchor_role": role,
            "event_id": expected_event,
            "event_timestamp": row.get("timestamp"),
            "event_actor": raw.get("actorID"),
            "event_object": raw.get("objectID"),
            "event_action": raw.get("action"),
            "reference_technique": technique,
            "reference_tactic": tactic,
            "valid_tactics": tactic_map[technique],
            "annotation_status": "final_high_confidence",
            "annotation_protocol": "official red-team log text + exact optc-labels event + PIDSMaker actor/object anchor; manual review",
            "source_corpus": "FiveDirections OpTC ground truth / optc-labels / PIDSMaker",
            "source_locator": {"task_id": task_id, "raw_event_id": expected_event},
            "source_record_sha256": source_record["record_sha256"],
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    source_records.sort(key=lambda row: (row["raw_event"]["timestamp"], row["task_id"]))
    evidence_path = output_dir / "source_records.jsonl"
    evidence_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in source_records),
        encoding="utf-8",
    )
    evidence_sha = _sha256(evidence_path)
    for row in draft_annotations:
        row.update(
            source_id=row["annotation_id"],
            source_record="source_records.jsonl",
            source_hash=evidence_sha,
        )
    draft_annotations.sort(key=lambda row: (row["event_timestamp"], row["annotation_id"]))
    annotations_path = output_dir / "source_linked_annotations.jsonl"
    annotations_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in draft_annotations),
        encoding="utf-8",
    )
    evidence_by_hash = {row["record_sha256"]: row for row in source_records}
    mapping_rows = []
    for row in draft_annotations:
        evidence = evidence_by_hash[row["source_record_sha256"]]
        raw = evidence["raw_event"]
        mapping_rows.append({
            "record_type": "mapping",
            "dataset": row["dataset"],
            "scene": row["scene"],
            "host_id": row["host_id"],
            "anchor_uuid": row["anchor_uuid"],
            "reference_technique": row["reference_technique"],
            "reference_tactic": row["reference_tactic"],
            "source_id": row["source_id"],
            "source_record": row["source_record"],
            "source_hash": row["source_hash"],
            "source_corpus": row["source_corpus"],
            "source_record_sha256": row["source_record_sha256"],
            "boundary": {
                "event_id": row["event_id"],
                "anchor": row["anchor_uuid"],
                "actor": row["event_actor"],
                "object": row["event_object"],
                "action": row["event_action"],
                "timestamp": row["event_timestamp"],
                "source_event_sha256": raw["source_record_sha256"],
            },
        })
    mapping_path = output_dir / "mapping_records.jsonl"
    mapping_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in mapping_rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "dataset": "optcday1",
        "annotation_count": len(draft_annotations),
        "annotation_status": "final_high_confidence",
        "source_repository": OPTC_LABELS_REPOSITORY,
        "source_commit": OPTC_LABELS_COMMIT,
        "source_archives": {
            "tasks/tasks.zip": TASKS_ZIP_SHA256,
            "labels/malicious.zip": MALICIOUS_ZIP_SHA256,
        },
        "attack_taxonomy": {
            "path": "data/attack_knowledge/mitre_attack/technique_to_tactic.json",
            "sha256": _sha256(tactic_path),
        },
        "inputs": {"candidate_jsonl_sha256": _sha256(candidates_path)},
        "outputs": {
            "source_records.jsonl": evidence_sha,
            "source_linked_annotations.jsonl": _sha256(annotations_path),
            "mapping_records.jsonl": _sha256(mapping_path),
        },
        "converter": {
            "path": "scripts/import_optc_attack_annotations.py",
            "sha256": _sha256(Path(__file__)),
        },
    }
    manifest["aggregate_sha256"] = _canonical_hash(manifest)
    (output_dir / "content_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "data/annotated_labels/optc/attack_techniques",
    )
    args = parser.parse_args(argv)
    print(json.dumps(build(args.candidates, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
