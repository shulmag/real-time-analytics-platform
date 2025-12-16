import React from 'react';
import { Navbar, Container, Nav } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import { auth } from '../services/auth';
import { signOut } from 'firebase/auth';

function NavBar() {
    const navigate = useNavigate();
    const userEmail = auth.currentUser?.email;

    const handleLogout = async () => {
        try {
            await signOut(auth);
            navigate('/login');
        } catch (error) {
            console.error('Failed to log out:', error);
        }
    };

    return (
        <Navbar bg="light" expand="lg">
            <Container fluid>
                <Navbar.Brand href="/prices">ficc.ai Muni Market Monitor</Navbar.Brand>
                <Navbar.Toggle aria-controls="basic-navbar-nav" />
                <Navbar.Collapse id="basic-navbar-nav">
                    <Nav className="me-auto">
                        <Nav.Link href="/prices">MUB Prices</Nav.Link>
                    </Nav>
                    <Navbar.Text className="me-3 text-muted">
                        Powered by S&P Data
                    </Navbar.Text>
                    <Navbar.Text className="me-3">
                        Signed in as: {userEmail}
                    </Navbar.Text>
                    <Nav>
                        <Nav.Link onClick={handleLogout}>Logout</Nav.Link>
                    </Nav>
                </Navbar.Collapse>
            </Container>
        </Navbar>
    );
}

export default NavBar;
