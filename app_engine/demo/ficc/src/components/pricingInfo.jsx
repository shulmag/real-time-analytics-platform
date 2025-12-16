import React from 'react';
import { Card as MuiCard, CardContent, Typography, Box, Button, List, ListItem, ListItemIcon, ListItemText, Divider, Link } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

const PricingCard = () => {
  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
      <Box sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', width: 400, mr: 2 }}>
        <MuiCard sx={{ borderRadius: '16px', boxShadow: 3, height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <CardContent>
            <Typography variant="h5" component="div" sx={{ mb: 2 }}>
              Web App or Data Vendor Package
            </Typography>
            <Typography variant="h4" component="div" sx={{ mb: 2, color: 'primary.main' }}>
              $5,000/month
            </Typography>
            <Typography variant="body1" component="div" sx={{ mb: 2 }}>
              Up to 5 users
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <List>
              <ListItem>
                <ListItemIcon>
                  <CheckCircleIcon sx={{ color: 'primary.main' }} />
                </ListItemIcon>
                <ListItemText primary="Accurate real-time pricing" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircleIcon sx={{ color: 'primary.main' }} />
                </ListItemIcon>
                <ListItemText primary="Updates every 1.3 seconds" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircleIcon sx={{ color: 'primary.main' }} />
                </ListItemIcon>
                <ListItemText primary="Considers trade size and side" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircleIcon sx={{ color: 'primary.main' }} />
                </ListItemIcon>
                <ListItemText primary="Access via web app or through data vendors" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircleIcon sx={{ color: 'primary.main' }} />
                </ListItemIcon>
                <ListItemText primary="24/7 customer support" />
              </ListItem>
            </List>
          </CardContent>
          <Box sx={{ textAlign: 'center', p: 3 }}>
            <Link href="https://pricing.ficc.ai/contact" underline="none">
              <Button variant="contained" color="primary">
                Get Started
              </Button>
            </Link>
          </Box>
        </MuiCard>
      </Box>

      <Box sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', width: 400 }}>
        <MuiCard sx={{ borderRadius: '16px', boxShadow: 3, height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <CardContent>
            <Typography variant="h5" component="div" sx={{ mb: 2 }}>
              API Access Package
            </Typography>
            <Typography variant="h4" component="div" sx={{ mb: 2, color: 'primary.main' }}>
              $10,000/month
            </Typography>
            <Typography variant="body1" component="div" sx={{ mb: 2 }}>
              Price millions of trades in real-time
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <List>
              <ListItem>
                <ListItemIcon>
                  <CheckCircleIcon sx={{ color: 'primary.main' }} />
                </ListItemIcon>
                <ListItemText primary="Flexible integration options" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircleIcon sx={{ color: 'primary.main' }} />
                </ListItemIcon>
                <ListItemText primary="Comprehensive documentation" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircleIcon sx={{ color: 'primary.main' }} />
                </ListItemIcon>
                <ListItemText primary="Dedicated support" />
              </ListItem>
            </List>
          </CardContent>
          <Box sx={{ textAlign: 'center', p: 3 }}>
            <Link href="https://pricing.ficc.ai/contact" underline="none">
              <Button variant="contained" color="primary">
                Get Started
              </Button>
            </Link>
          </Box>
        </MuiCard>
      </Box>
    </Box>
  );
};

export default PricingCard;
