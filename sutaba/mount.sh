DEVICE_ID=sdb
MNT_DIR=sutaba
# sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/$DEVICE_ID
sudo mkdir -p /mnt/disks/$MNT_DIR
sudo mount -o discard,defaults /dev/$DEVICE_ID /mnt/disks/$MNT_DIR
sudo chmod a+w /mnt/disks/$MNT_DIR
sudo cp /etc/fstab /etc/fstab.backup
sudo blkid /dev/$MNT_DIR
echo UUID=`sudo blkid -s UUID -o value /dev/$DEVICE_ID` /mnt/disks/sutaba ext4 discard,defaults,nofail 0 2 | sudo tee -a /etc/fstab