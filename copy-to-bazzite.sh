#!/bin/bash
rsync -arzhv0 ./barq_images/ user@bazzite:/media/user/SSD2TB/rutorch2/pursuit/barq_images
rsync -arzhv0 ./barq_cache.db* user@bazzite:/media/user/SSD2TB/rutorch2/pursuit/
rsync -arzhv0 ./datasets/ user@bazzite:/media/user/SSD2TB/rutorch2/pursuit/datasets

