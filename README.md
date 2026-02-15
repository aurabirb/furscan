# Pursuit - Fursuit Character Recognition
Available on Telegram as [@furscanbot](https://t.me/furscanbot) and other names.

## Read [CLAUDE.md](CLAUDE.md) for a description of this project

## TODO:
- [ ] Create a periodic schedule job to download and ingest new images from public sources
- [ ] Download nfc26 dataset
- [ ] Csontact soragryphon about integrating this to furtrack
- [ ] Fix the nsfw filter
- [ ] Number the segments on the drawn label
- [ ] Rename the confusing "segment" naming when replying to the user, instead use "fursuit"
- [ ] Create a periodic job to backup datasets & databases
- [ ] User interface for submitting new pictures should be nice and fun to use
- [ ] Make a webapp for in-browser fursuit identification (e.g. tag all images from my camera sdcard)
- [ ] Create an incentive for user data submission (game, find your fursuit parents, lookalikes, scores? etc) - make it beneficial or interesting to use
- [ ] Prioritize most recently seen fursuits when scoring results
- [ ] Allow users to say "@bot this is character_name" and "@bot this not character_name"
- [ ] Create nice icons for tg bots (I'm thinking of the fursuit with a labeled bounding box, so that it is obvious what this bot does)
- [ ] Call to action to submit your own pictures to the database
- [ ] Parse text (e.g. badges) and add to the search database
- [ ] Document how to set up the bot in group chats
- [ ] Make the bot respond to edited messages
- [ ] Add a link to the webapp to the bot hello message.
- [ ] Create call to action to upload some of the pictures you took to furtrack? Labels optional because I don't want to pollute furtrack with bad labels just yet. This will hurt training.
- [ ] Find your own pictures in the database, even if they are not uploaded to furtrack?
- [ ] Try to make it possible to tag several people in the picture correctly (left to right?)
- [ ] Import an alias database from furtrack so that we can cross-validate characters appearing in e.g. nfc25 and furtrack
- [ ] Deprioritize low quality segments (low confidence, low res, unusual aspect crop for the prompt etc)
- [ ] Deprioritize low frequency or very old fursuits?
- [ ] Overlapping segments - can we resolve them? Esp. relevant for manual tagging from left to right
- [ ] Find pictures with multiple fursuits and if we have at least 3, we can pick out which segment is the real tag by running self-similarity on a character pictures and mark all other segments as someone else. That way we reduce the noise.
- [ ] Run a self-similarity search on all database and cluster all potential segments to potentially assign it a tag, this is an extension of the previous point.
- [ ] Identify not just the "fursuiter head" but the whole body
- [ ] Create other preprocessing pipelines and assess their score on the validation dataset, such as black-and-white preprocessing, brightness normalization, etc.
- [ ] Make a feature to import the index data from another instance (to make it sync new fursuits across several running instances of the detector)
- [ ] Double-check if the /show command actually does not leak the user-submitted pictures from telegram
- [ ] Create an app that finds fursuit pictures in the camera roll (and maybe monitors it) and keeps a list of who you took pictures of
- [ ] Add social login to the bot hello message, ie login with google/furaffinity/furtrack/barq/twitter etc so that we can analyze, attribute and upload images there later
- [ ] Use Filesystem API in the webapp to list contents of users' folder periodically without having to select individual photos.
- [ ] Add mode to sync the index and database periodically to an upstream S3 bucket, not sure how to do that exactly but maybe shard them into append-only pieces (kinda like wal or wal itself) and upload those periodically, and merge on the client?
- [ ] Use image infill to add occluded part of the fursuit head
- [ ] Add the full fursuit scanner to index on other parts of the body
- [ ] Use e.g. depth-anything to generate extra angles on the fursuit
- [ ] Run clip on the fursuit crop and create a text search index e.g. /search blue fox (done but not working)
- [ ] Run a SAM2 + clip on each fragment instead of heavy SAM3 to segment where the fursuit head is.
