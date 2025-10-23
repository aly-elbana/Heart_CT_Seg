# class UNet(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.f = 64

#         self.conv1_1 = nn.Conv2d(1, self.f, 3, padding=1)
#         self.conv1_2 = nn.Conv2d(self.f, self.f, 3, padding=1)

#         self.conv2_1 = nn.Conv2d(self.f, self.f * 2, 3, padding=1)
#         self.conv2_2 = nn.Conv2d(self.f * 2, self.f * 2, 3, padding=1)

#         self.conv3_1 = nn.Conv2d(self.f * 2, self.f * 4, 3, padding=1)
#         self.conv3_2 = nn.Conv2d(self.f * 4, self.f * 4, 3, padding=1)

#         self.conv4_1 = nn.Conv2d(self.f * 4, self.f * 8, 3, padding=1)
#         self.conv4_2 = nn.Conv2d(self.f * 8, self.f * 8, 3, padding=1)

#         self.pool = nn.MaxPool2d(2)
#         self.dropout = nn.Dropout(0.3)



#         self.bottleneck1 = nn.Conv2d(self.f * 8, self.f * 16, 3, padding=1)
#         self.bottleneck2 = nn.Conv2d(self.f * 16, self.f * 16, 3, padding=1)



#         self.upconv4 = nn.ConvTranspose2d(self.f * 16, self.f * 8, 2, stride=2)
#         self.dec4_1 = nn.Conv2d(self.f * 16, self.f * 8, 3, padding=1)
#         self.dec4_2 = nn.Conv2d(self.f * 8, self.f * 8, 3, padding=1)

#         self.upconv3 = nn.ConvTranspose2d(self.f * 8, self.f * 4, 2, stride=2)
#         self.dec3_1 = nn.Conv2d(self.f * 8, self.f * 4, 3, padding=1)
#         self.dec3_2 = nn.Conv2d(self.f * 4, self.f * 4, 3, padding=1)

#         self.upconv2 = nn.ConvTranspose2d(self.f * 4, self.f * 2, 2, stride=2)
#         self.dec2_1 = nn.Conv2d(self.f * 4, self.f * 2, 3, padding=1)
#         self.dec2_2 = nn.Conv2d(self.f * 2, self.f * 2, 3, padding=1)

#         self.upconv1 = nn.ConvTranspose2d(self.f * 2, self.f, 2, stride=2)
#         self.dec1_1 = nn.Conv2d(self.f * 2, self.f, 3, padding=1)
#         self.dec1_2 = nn.Conv2d(self.f, self.f, 3, padding=1)



#         self.final = nn.Conv2d(self.f, 1, 1)

#     def forward(self, x):
#         c1 = F.relu(self.conv1_1(x))
#         c1 = F.relu(self.conv1_2(c1))
#         p1 = self.pool(c1)

#         c2 = F.relu(self.conv2_1(p1))
#         c2 = F.relu(self.conv2_2(c2))
#         p2 = self.pool(c2)

#         c3 = F.relu(self.conv3_1(p2))
#         c3 = F.relu(self.conv3_2(c3))
#         p3 = self.pool(c3)

#         c4 = F.relu(self.conv4_1(p3))
#         c4 = F.relu(self.conv4_2(c4))
#         p4 = self.pool(c4)



#         bn = F.relu(self.bottleneck1(p4))
#         bn = F.relu(self.bottleneck2(bn))



#         u4 = self.upconv4(bn)
#         u4 = torch.cat([u4, c4], dim=1)
#         u4 = F.relu(self.dec4_1(u4))
#         u4 = F.relu(self.dec4_2(u4))

#         u3 = self.upconv3(u4)
#         u3 = torch.cat([u3, c3], dim=1)
#         u3 = F.relu(self.dec3_1(u3))
#         u3 = F.relu(self.dec3_2(u3))

#         u2 = self.upconv2(u3)
#         u2 = torch.cat([u2, c2], dim=1)
#         u2 = F.relu(self.dec2_1(u2))
#         u2 = F.relu(self.dec2_2(u2))

#         u1 = self.upconv1(u2)
#         u1 = torch.cat([u1, c1], dim=1)
#         u1 = F.relu(self.dec1_1(u1))
#         u1 = F.relu(self.dec1_2(u1))



#         out = self.final(u1)
#         out = torch.sigmoid(out)
#         return out
