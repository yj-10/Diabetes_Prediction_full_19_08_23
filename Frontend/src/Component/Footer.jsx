import React from "react";
import { Col, Container, Row } from "react-bootstrap";
import "../style/Footer.css";
import infosense from "../assest/Infosense.png";
import { FiFacebook, FiTwitter, FiInstagram, FiLinkedin } from "react-icons/fi";
function Footer() {
  return (
    <>
      {/* <div className="footer_wrapper">
        <Container>
          <Row className="d-flex justify-contnet-between align-items-center">
            <Col lg={8}>
              <img src={infosense} className="img-fluid w-50" alt="no-img" />
            </Col>

            <Col lg={4}>
              <div className="footer_wrapper_content">
                <ul>
                  <li>info@infosense.tech</li>
                  <li>Contact Us +91 - 9343017165</li>
                </ul>
              </div>
            </Col>
          </Row>
        </Container>
      </div> */}
      <div className="footer_container">
        <Container>
          <Row className="d-flex justify-contnet-between align-items-center">
            <Col sm={12} md={6} lg={6}>
              <div className="footer_content text-capitalize">
                Copyright © Infosense ai Pvt. Ltd. {new Date().getFullYear()}{" "}
                All rights reserved
              </div>
            </Col>

            <Col sm={12} md={6} lg={6}>
              <ul>
                <li>
                  <a href="">
                    <FiFacebook />
                  </a>
                </li>
                <li>
                  <a href="">
                    <FiInstagram />
                  </a>
                </li>
                <li>
                  <a href="">
                    <FiLinkedin />
                  </a>
                </li>
                <li>
                  <a href="">
                    <FiTwitter />
                  </a>
                </li>
              </ul>
            </Col>
          </Row>
        </Container>
      </div>
    </>
  );
}

export default Footer;
