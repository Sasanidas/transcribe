# Google Docs setup (one time, ~10 minutes)

The transcriber writes to Google Docs through your own Google Cloud project. It asks only for
the `drive.file` permission: it can create documents and edit **only the documents it created**.
It cannot see or change anything else in your Drive.

Google's console changes its menu names from time to time; the steps below match the
"Google Auth Platform" layout.

## 1. Create a project and enable the Docs API
1. Open <https://console.cloud.google.com/> and sign in with the Google account whose Drive
   should hold the transcripts.
2. Project picker (top bar) → **New project** → name it e.g. `meet-transcriber` → **Create**,
   then make sure it is selected.
3. **APIs & Services → Library** → search **Google Docs API** → **Enable**.

## 2. Configure the OAuth consent screen
1. **APIs & Services → OAuth consent screen** (opens *Google Auth Platform*) → **Get started**.
2. App name: `Meet Transcriber`; user support email: yours → **Next**.
3. Audience: **Internal** if you have a Google Workspace (company) account and the option is
   offered, otherwise **External** → **Next**.
4. Contact email: yours → **Next** → accept the policy → **Create**.
5. If you chose External: **Audience → Test users → Add users** → add your own Google address.
6. **Data access → Add or remove scopes** → filter for `drive.file` → tick
   `.../auth/drive.file` → **Update** → **Save**.

## 3. Create the desktop client
1. **Clients → Create client** → Application type **Desktop app** → name `transcriber` → **Create**.
2. **Download JSON** and save it as:
   `~/Programming/transcribe/credentials.json`
   (it is listed in `.gitignore`; don't share it).

## 4. Log in once
```
cd ~/Programming/transcribe
uv run python -m transcribe.gdoc --login
```
A browser tab opens. Choose your account and allow access. With an External app in testing mode,
Google shows "Google hasn't verified this app" → **Continue** (it is your own app).
The login is saved in `token.json` (also git-ignored, readable only by you).

Check that it works. This creates a small test document and prints its link:
```
uv run python -m transcribe.gdoc --test
```

## 5. Use it
```
uv run python -m transcribe.live --doc                        # new doc per meeting
uv run python -m transcribe.live --doc --doc-title "Weekly sync 11/18"
uv run python -m transcribe.live --doc-id <id>                # keep appending to one doc
```
The doc link is printed at start. Lines appear in batches every ~3 s. Renames (`2=Tanaka`,
`me=Your Name`) also update lines already in the doc.

## Notes
- **Weekly re-login (External apps only):** while an External app is in *Testing* status,
  Google expires the login after 7 days, so you'd run `--login` again. To avoid that, set
  **Audience → Publishing status → Publish app**. Since `drive.file` is a non-sensitive scope,
  no Google review is needed for your own use. Internal apps don't have this limit.
- `--doc-id` only works for documents this app created (that's the `drive.file` limit).
- If Google is unreachable, lines are retried; the local transcript in `transcripts/` always
  has everything.
